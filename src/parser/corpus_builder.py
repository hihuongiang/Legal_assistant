"""Build a validated effective-date legal corpus from raw Parquet documents."""

from __future__ import annotations

from dataclasses import asdict
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


def _parse_iso_date(value: object, field_name: str) -> date:
    if not isinstance(value, str):
        raise CorpusValidationError(f"{field_name} must use YYYY-MM-DD, got {value!r}")
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise CorpusValidationError(
            f"{field_name} must use YYYY-MM-DD, got {value!r}"
        ) from error


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
    parsed_dates = [
        _parse_iso_date(value, f"effFrom at source row {index}")
        for index, value in selected["effFrom"].items()
    ]
    selected["_effective_date"] = parsed_dates
    eligible = selected[
        (selected["_effective_date"] <= effective_as_of)
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
        columns=["_source_order", "_effective_date", "_has_html", "_html_length"]
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
    """Create paragraph and sentence units, deferring word breaks to the packer."""
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
        _parse_iso_date(effective_date, "effFrom")
        status = _required_row_text(row, "status")
        html_content = _required_row_text(row, "html_content")

        articles: list[tuple[int, list[str]]] = []
        current_article: int | None = None
        current_lines: list[str] = []
        for line in _html_lines(html_content):
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
            raise CorpusValidationError("document produced no article chunks")

        chunks: list[LegalChunk] = []
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
                )
            )

        duplicate_ids = {chunk.chunk_id for chunk in chunks if sum(c.chunk_id == chunk.chunk_id for c in chunks) > 1}
        if duplicate_ids:
            raise CorpusValidationError(f"duplicate chunk IDs: {', '.join(sorted(duplicate_ids))}")
        return chunks

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
            base_id = f"{doc_id}_D{article_number}_K{clause_number}"
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
    documents = select_effective_documents(pd.read_parquet(source_path), effective_as_of)
    chunker = ArticleChunker()
    chunks = [chunk for _, row in documents.iterrows() for chunk in chunker.chunk_document(row)]
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise CorpusValidationError("duplicate chunk IDs in effective corpus")

    chunk_content = (
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    manifest = CorpusBuildManifest(
        as_of_date=effective_as_of.isoformat(),
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        document_count=len(documents),
        chunk_count=len(chunks),
        corpus_sha256=hashlib.sha256(chunk_content).hexdigest(),
    )
    output_path = Path(output_dir)
    _write_atomically(output_path / "effective_legal_chunks.json", chunk_content)
    _write_atomically(
        output_path / "effective_legal_corpus.manifest.json",
        (json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return manifest
