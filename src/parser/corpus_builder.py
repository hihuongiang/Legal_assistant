"""Build a validated effective-date legal corpus from raw Parquet documents."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from bs4 import BeautifulSoup
import pandas as pd

from src.parser.models import CorpusBuildManifest, CorpusValidationError, LegalChunk


REQUIRED_SOURCE_FIELDS = (
    "docs_code",
    "docs_title",
    "source_url",
    "issue_date",
    "effFrom",
    "status",
    "html_content",
)
EXCLUDED_STATUSES = {"Hết hiệu lực", "Bị bãi bỏ", "Ngưng hiệu lực", "Đình chỉ"}
MAX_CONTENT_WORDS = 350
OVERLAP_WORDS = 40
MAX_PERSISTED_WORDS = MAX_CONTENT_WORDS + OVERLAP_WORDS
MISSING_EFFECTIVE_DATE_REASON = "unknown legal effective date"
DUPLICATE_CLAUSE_OCCURRENCE_PATTERN = re.compile(r"_O\d+(?:_P\d+)?$")


def _parse_iso_date(value: object, field_name: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise CorpusValidationError(f"{field_name} must use YYYY-MM-DD, got {value!r}")
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise CorpusValidationError(
            f"{field_name} must use YYYY-MM-DD, got {value!r}"
        ) from error


def _parse_effective_interval(value: object, field_name: str) -> tuple[date, date]:
    """Return the known effective-date interval without inventing a day for year-only input."""
    if isinstance(value, str) and re.fullmatch(r"\d{4}", value):
        try:
            year = int(value)
            return date(year, 1, 1), date(year, 12, 31)
        except ValueError as error:
            raise CorpusValidationError(
                f"{field_name} must use YYYY or YYYY-MM-DD, got {value!r}"
            ) from error

    exact_date = _parse_iso_date(value, field_name)
    return exact_date, exact_date


def _has_missing_effective_date(value: object) -> bool:
    """Return whether the raw source provides no legal effective-date value."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return bool(pd.isna(value))


def count_missing_effective_dates(frame: pd.DataFrame) -> int:
    """Count source records excluded because their legal effective date is unknown."""
    if "effFrom" not in frame.columns:
        return 0
    return sum(_has_missing_effective_date(value) for value in frame["effFrom"])


def _selection_audit_counts(frame: pd.DataFrame, as_of: date) -> dict[str, int]:
    """Classify each raw row once so the persisted manifest reconciles its input."""
    missing_mask = frame["effFrom"].map(_has_missing_effective_date)
    nonmissing = frame.loc[~missing_mask]
    effective_intervals = {
        index: _parse_effective_interval(value, f"effFrom at source row {index}")
        for index, value in nonmissing["effFrom"].items()
    }
    ambiguous_row = next(
        (
            index
            for index, (start, end) in effective_intervals.items()
            if start != end and start <= as_of <= end
        ),
        None,
    )
    if ambiguous_row is not None:
        raise CorpusValidationError(
            "as_of falls inside year-only effFrom interval "
            f"at source row {ambiguous_row}: {nonmissing.at[ambiguous_row, 'effFrom']!r}"
        )

    future_mask = nonmissing.index.to_series().map(
        lambda index: effective_intervals[index][1] > as_of
    )
    inactive_mask = ~future_mask & nonmissing["status"].isin(EXCLUDED_STATUSES)
    return {
        "raw_document_count": len(frame),
        "excluded_missing_effective_date_count": int(missing_mask.sum()),
        "excluded_future_effective_date_count": int(future_mask.sum()),
        "excluded_inactive_status_count": int(inactive_mask.sum()),
        "eligible_document_count": int((~future_mask & ~inactive_mask).sum()),
    }


def count_duplicate_clause_occurrences(chunks: list[LegalChunk]) -> int:
    """Count repeated clause occurrences once even when one is split into multiple chunks."""
    occurrence_ids = {
        re.sub(r"_P\d+$", "", chunk.chunk_id)
        for chunk in chunks
        if DUPLICATE_CLAUSE_OCCURRENCE_PATTERN.search(chunk.chunk_id)
    }
    return len(occurrence_ids)


def _chunk_id_occurrence_parts(chunk_id: str) -> tuple[str, str, str]:
    """Return canonical base, source occurrence base, and split-part suffix."""
    part_match = re.search(r"_P\d+$", chunk_id)
    part_suffix = part_match.group(0) if part_match else ""
    source_base = chunk_id[: -len(part_suffix)] if part_suffix else chunk_id
    canonical_base = re.sub(r"_O\d+$", "", source_base)
    return canonical_base, source_base, part_suffix


def _suffix_duplicate_chunk_ids(
    chunks: list[LegalChunk], occurrence_counts: dict[str, int]
) -> tuple[list[LegalChunk], int]:
    """Make normalized IDs unique while preserving source order and split parts."""
    source_occurrences: dict[str, int] = {}
    duplicate_clause_count = 0
    suffixed: list[LegalChunk] = []
    for chunk in chunks:
        canonical_base, source_base, part_suffix = _chunk_id_occurrence_parts(chunk.chunk_id)
        if source_base not in source_occurrences:
            occurrence = occurrence_counts.get(canonical_base, 0) + 1
            occurrence_counts[canonical_base] = occurrence
            source_occurrences[source_base] = occurrence
            if occurrence > 1 and re.search(r"_D\d+_K\d+$", canonical_base):
                duplicate_clause_count += 1
        occurrence = source_occurrences[source_base]
        occurrence_suffix = f"_O{occurrence}" if occurrence > 1 else ""
        suffixed.append(replace(chunk, chunk_id=f"{canonical_base}{occurrence_suffix}{part_suffix}"))
    return suffixed, duplicate_clause_count


def _parse_as_of(as_of: str | date) -> date:
    if isinstance(as_of, datetime):
        return as_of.date()
    if isinstance(as_of, date):
        return as_of
    return _parse_iso_date(as_of, "as_of")


def select_effective_documents(frame: pd.DataFrame, as_of: str | date) -> pd.DataFrame:
    """Return one eligible, best raw version of each legal document code."""
    missing_fields = [field for field in REQUIRED_SOURCE_FIELDS if field not in frame.columns]
    if missing_fields:
        raise CorpusValidationError(
            f"raw source is missing required field(s): {', '.join(missing_fields)}"
        )

    effective_as_of = _parse_as_of(as_of)
    selected = frame.copy().reset_index(drop=True)
    selected["_source_order"] = range(len(selected))
    selected = selected.loc[
        ~selected["effFrom"].map(_has_missing_effective_date)
    ].copy()
    effective_intervals = {
        index: _parse_effective_interval(value, f"effFrom at source row {index}")
        for index, value in selected["effFrom"].items()
    }
    ambiguous_row = next(
        (
            index
            for index, (start, end) in effective_intervals.items()
            if start != end and start <= effective_as_of <= end
        ),
        None,
    )
    if ambiguous_row is not None:
        raise CorpusValidationError(
            "as_of falls inside year-only effFrom interval "
            f"at source row {ambiguous_row}: {selected.at[ambiguous_row, 'effFrom']!r}"
        )

    selected["_effective_start"] = [
        effective_intervals[index][0] for index in selected.index
    ]
    selected["_effective_end"] = [
        effective_intervals[index][1] for index in selected.index
    ]
    eligible = selected[
        (selected["_effective_end"] <= effective_as_of)
        & ~selected["status"].isin(EXCLUDED_STATUSES)
    ].copy()

    html = eligible["html_content"].fillna("").astype(str)
    eligible["_has_html"] = html.str.strip().ne("")
    eligible["_html_length"] = html.str.len()
    ranked = eligible.sort_values(
        ["docs_code", "_has_html", "_html_length", "_source_order"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    winners = ranked.drop_duplicates(subset=["docs_code"], keep="first")
    return winners.sort_values("_source_order", kind="stable").drop(
        columns=[
            "_source_order",
            "_effective_start",
            "_effective_end",
            "_has_html",
            "_html_length",
        ]
    ).reset_index(drop=True)


def _normalize_doc_code(doc_code: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", doc_code.strip()).strip("-").upper()
    if not normalized:
        raise CorpusValidationError("docs_code must be a non-empty string")
    return normalized


def _required_row_text(row: Mapping[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise CorpusValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _html_lines(html_content: str) -> list[str]:
    soup = BeautifulSoup(html_content, "html.parser")
    for hidden in soup.find_all(attrs={"style": re.compile(r"display\s*:\s*none", re.I)}):
        hidden.decompose()
    return [
        re.sub(r"\s+", " ", line).strip()
        for line in soup.get_text(separator="\n").splitlines()
        if line.strip()
    ]


def _content_units(lines: list[str]) -> list[str]:
    """Create complete paragraph units, then sentences for oversize paragraphs."""
    units: list[str] = []
    for line in lines:
        if len(line.split()) <= MAX_CONTENT_WORDS:
            units.append(line)
            continue
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", line)
            if sentence.strip()
        ]
        units.extend(sentences)
    return units


def _split_content(lines: list[str]) -> list[str]:
    units = _content_units(lines)
    core_chunks: list[list[str]] = []
    current: list[str] = []
    current_word_count = 0
    for unit in units:
        unit_words = unit.split()
        if len(unit_words) <= MAX_CONTENT_WORDS:
            if current and current_word_count + len(unit_words) > MAX_CONTENT_WORDS:
                core_chunks.append(current)
                current = []
                current_word_count = 0
            current.extend(unit_words)
            current_word_count += len(unit_words)
            continue

        while unit_words:
            available_words = MAX_CONTENT_WORDS - current_word_count
            current.extend(unit_words[:available_words])
            current_word_count += min(len(unit_words), available_words)
            unit_words = unit_words[available_words:]
            if current_word_count == MAX_CONTENT_WORDS:
                core_chunks.append(current)
                current = []
                current_word_count = 0
    if current:
        core_chunks.append(current)

    chunks: list[str] = []
    prior_words: list[str] = []
    for core in core_chunks:
        words = " ".join(core).split()
        if prior_words:
            words = prior_words[-OVERLAP_WORDS:] + words
        content = " ".join(words).strip()
        if not content or len(words) > MAX_PERSISTED_WORDS:
            raise CorpusValidationError("chunk content is empty or exceeds 390 words")
        chunks.append(content)
        prior_words = words
    return chunks


class ArticleChunker:
    """Convert one selected raw document into canonical article/clause chunks."""

    article_pattern = re.compile(r"^(?:Article|Điều)\s+(\d+)(?:[.\s]|$)", re.I)
    clause_pattern = re.compile(r"^(\d+)\.\s+", re.I)
    numeric_line_pattern = re.compile(r"[0-9./\- ]+")

    def chunk_document(self, row: Mapping[str, Any]) -> list[LegalChunk]:
        doc_code = _required_row_text(row, "docs_code")
        doc_id = _normalize_doc_code(doc_code)
        law_name = _required_row_text(row, "docs_title")
        source_url = _required_row_text(row, "source_url")
        effective_date = _required_row_text(row, "effFrom")
        _parse_effective_interval(effective_date, "effFrom")
        status = _required_row_text(row, "status")
        html_content = _required_row_text(row, "html_content")

        document_lines = _html_lines(html_content)
        articles: list[tuple[int, list[str]]] = []
        current_article: int | None = None
        current_lines: list[str] = []
        for line in document_lines:
            match = self.article_pattern.match(line)
            if match:
                if current_article is not None:
                    articles.append((current_article, current_lines))
                current_article = int(match.group(1))
                current_lines = [line]
            elif current_article is not None:
                current_lines.append(line)
        if current_article is not None:
            articles.append((current_article, current_lines))
        if not articles:
            return self._chunk_full_document(
                doc_id=doc_id,
                doc_code=doc_code,
                law_name=law_name,
                source_url=source_url,
                effective_date=effective_date,
                status=status,
                lines=document_lines,
            )

        chunks: list[LegalChunk] = []
        clause_occurrences: dict[tuple[int, int], int] = {}
        for article_number, article_lines in articles:
            chunks.extend(
                self._chunk_article(
                    doc_id=doc_id,
                    doc_code=doc_code,
                    law_name=law_name,
                    source_url=source_url,
                    effective_date=effective_date,
                    status=status,
                    article_number=article_number,
                    lines=article_lines,
                    clause_occurrences=clause_occurrences,
                )
            )

        duplicate_ids = {chunk.chunk_id for chunk in chunks if sum(c.chunk_id == chunk.chunk_id for c in chunks) > 1}
        if duplicate_ids:
            raise CorpusValidationError(f"duplicate chunk IDs: {', '.join(sorted(duplicate_ids))}")
        return chunks

    def _chunk_full_document(
        self,
        *,
        doc_id: str,
        doc_code: str,
        law_name: str,
        source_url: str,
        effective_date: str,
        status: str,
        lines: list[str],
    ) -> list[LegalChunk]:
        """Chunk an effective document with no canonical article headings without inventing one."""
        parts = _split_content(lines)
        if not parts:
            raise CorpusValidationError("document produced no chunkable content")
        return [
            LegalChunk(
                chunk_id=f"{doc_id}_FULL_P{part_number}",
                law_id=doc_code,
                law_name=law_name,
                article_name="Toàn văn",
                clause_name="",
                effective_date=effective_date,
                status=status,
                source_url=source_url,
                content=content,
            )
            for part_number, content in enumerate(parts, start=1)
        ]

    def _chunk_article(
        self,
        *,
        doc_id: str,
        doc_code: str,
        law_name: str,
        source_url: str,
        effective_date: str,
        status: str,
        article_number: int,
        lines: list[str],
        clause_occurrences: dict[tuple[int, int], int],
    ) -> list[LegalChunk]:
        article_name = lines[0]
        header_lines = [lines[0]]
        clauses: list[tuple[int, list[str]]] = []
        current_clause_number: int | None = None
        current_clause_lines: list[str] = []
        numeric_run = 0
        for line in lines[1:]:
            if self.numeric_line_pattern.fullmatch(line):
                numeric_run += 1
                if numeric_run >= 30:
                    break
            else:
                numeric_run = 0

            clause_match = self.clause_pattern.match(line)
            if clause_match:
                if current_clause_number is not None:
                    clauses.append((current_clause_number, current_clause_lines))
                current_clause_number = int(clause_match.group(1))
                current_clause_lines = [line]
            elif current_clause_number is None:
                header_lines.append(line)
            else:
                current_clause_lines.append(line)
        if current_clause_number is not None:
            clauses.append((current_clause_number, current_clause_lines))
        if not clauses:
            clauses = [(1, header_lines)]

        chunks: list[LegalChunk] = []
        for clause_number, clause_lines in clauses:
            parts = _split_content(clause_lines)
            if not parts:
                continue
            occurrence_key = (article_number, clause_number)
            occurrence = clause_occurrences.get(occurrence_key, 0) + 1
            clause_occurrences[occurrence_key] = occurrence
            occurrence_suffix = f"_O{occurrence}" if occurrence > 1 else ""
            base_id = f"{doc_id}_D{article_number}_K{clause_number}{occurrence_suffix}"
            for part_number, content in enumerate(parts, start=1):
                suffix = f"_P{part_number}" if len(parts) > 1 else ""
                chunks.append(
                    LegalChunk(
                        chunk_id=f"{base_id}{suffix}",
                        law_id=doc_code,
                        law_name=law_name,
                        article_name=article_name,
                        clause_name=f"Clause {clause_number}",
                        effective_date=effective_date,
                        status=status,
                        source_url=source_url,
                        content=content,
                    )
                )
        return chunks


def _write_atomically(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary_file:
            temporary_path = temporary_file.name
            temporary_file.write(content)
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def build_effective_corpus(
    source: str | Path, output_dir: str | Path, as_of: str | date
) -> CorpusBuildManifest:
    """Build effective legal chunks from a raw Parquet source and persist both artifacts."""
    source_path = Path(source)
    source_bytes = source_path.read_bytes()
    effective_as_of = _parse_as_of(as_of)
    raw_documents = pd.read_parquet(source_path)
    missing_fields = [field for field in REQUIRED_SOURCE_FIELDS if field not in raw_documents.columns]
    if missing_fields:
        raise CorpusValidationError(
            f"raw source is missing required field(s): {', '.join(missing_fields)}"
        )
    audit_counts = _selection_audit_counts(raw_documents, effective_as_of)
    documents = select_effective_documents(raw_documents, effective_as_of)
    audit_counts["excluded_duplicate_document_count"] = (
        audit_counts["eligible_document_count"] - len(documents)
    )
    chunker = ArticleChunker()
    chunks: list[LegalChunk] = []
    fallback_document_count = 0
    fallback_chunk_count = 0
    duplicate_clause_occurrence_count = 0
    occurrence_counts: dict[str, int] = {}
    for _, row in documents.iterrows():
        document_chunks = chunker.chunk_document(row)
        document_chunks, duplicate_count = _suffix_duplicate_chunk_ids(
            document_chunks, occurrence_counts
        )
        fallback_chunks = [chunk for chunk in document_chunks if "_FULL_P" in chunk.chunk_id]
        if fallback_chunks:
            fallback_document_count += 1
            fallback_chunk_count += len(fallback_chunks)
        duplicate_clause_occurrence_count += duplicate_count
        chunks.extend(document_chunks)
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise CorpusValidationError("duplicate chunk IDs in effective corpus")

    chunk_content = (
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    manifest = CorpusBuildManifest(
        as_of_date=effective_as_of.isoformat(),
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        raw_document_count=audit_counts["raw_document_count"],
        eligible_document_count=audit_counts["eligible_document_count"],
        document_count=len(documents),
        chunk_count=len(chunks),
        excluded_missing_effective_date_count=audit_counts["excluded_missing_effective_date_count"],
        excluded_missing_effective_date_reason=MISSING_EFFECTIVE_DATE_REASON,
        excluded_future_effective_date_count=audit_counts["excluded_future_effective_date_count"],
        excluded_inactive_status_count=audit_counts["excluded_inactive_status_count"],
        excluded_duplicate_document_count=audit_counts["excluded_duplicate_document_count"],
        fallback_document_count=fallback_document_count,
        fallback_chunk_count=fallback_chunk_count,
        duplicate_clause_occurrence_count=duplicate_clause_occurrence_count,
        corpus_sha256=hashlib.sha256(chunk_content).hexdigest(),
    )
    output_path = Path(output_dir)
    _write_atomically(output_path / "effective_legal_chunks.json", chunk_content)
    _write_atomically(
        output_path / "effective_legal_corpus.manifest.json",
        (json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return manifest
