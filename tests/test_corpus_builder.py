import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import date

import pandas as pd
import pytest

from src.parser.models import CorpusValidationError


def document(**overrides: object) -> dict[str, object]:
    """Return one complete raw-document row with stable legal metadata."""
    row: dict[str, object] = {
        "docs_code": "45/2026/QH",
        "docs_title": "Law on Corpus Fixtures",
        "source_url": "https://example.test/law-45-2026",
        "issue_date": "2026-01-01",
        "effFrom": "2026-07-01",
        "status": "Chưa có hiệu lực",
        "html_content": (
            "<article><p>Article 1. Scope</p><p>1. This is the first clause.</p>"
            "<p>a) A point in the first clause.</p><p>2. This is the second clause.</p>"
            "</article>"
        ),
    }
    row.update(overrides)
    return row


@pytest.fixture
def documents_with_missing_effective_date() -> pd.DataFrame:
    """Raw rows that distinguish an unknown date from a future or stale status."""
    return pd.DataFrame(
        [
            document(docs_code="INCLUDED"),
            document(docs_code="UNKNOWN", effFrom="", status="active"),
        ]
    )


@pytest.fixture
def section_only_document() -> dict[str, object]:
    """An effective document with prose sections but no canonical Article heading."""
    return document(
        docs_code="SECTION-ONLY",
        html_content=(
            "<article><p>PHẦN I. QUY ĐỊNH CHUNG</p>"
            "<p>Nội dung toàn văn không có tiêu đề Điều được đánh số.</p></article>"
        ),
    )


@pytest.fixture
def duplicate_clause_document() -> dict[str, object]:
    """A legal article whose embedded amendment repeats a canonical clause number."""
    return document(
        docs_code="DUPLICATE-CLAUSE",
        html_content=(
            "<article><p>Article 4. Amending provisions</p>"
            "<p>1. First top-level clause.</p><p>1. Embedded amended clause.</p>"
            "<p>2. Unchanged clause.</p></article>"
        ),
    )


@pytest.fixture
def duplicate_article_document() -> dict[str, object]:
    """A crawled document that repeats one canonical article and its clauses."""
    return document(
        docs_code="DUPLICATE-ARTICLE",
        html_content=(
            "<article><p>Article 4. Amending provisions</p>"
            "<p>1. First article occurrence.</p></article>"
            "<article><p>Article 4. Amending provisions (repeated)</p>"
            "<p>1. Second article occurrence.</p></article>"
        ),
    )


def test_select_effective_documents_filters_statuses_dates_and_deduplicates():
    """Catches an eligibility filter that drops pending-but-effective laws or keeps invalid duplicates."""
    from src.parser.corpus_builder import select_effective_documents

    frame = pd.DataFrame(
        [
            document(docs_code="KEEP", html_content="<p>short</p>"),
            document(docs_code="KEEP", html_content=""),
            document(docs_code="KEEP", html_content="<p>the longest retained version</p>"),
            document(docs_code="FUTURE", effFrom="2026-09-01"),
            document(docs_code="EXPIRED", status="Hết hiệu lực"),
            document(docs_code="REPEALED", status="Bị bãi bỏ"),
            document(docs_code="SUSPENDED", status="Ngưng hiệu lực"),
            document(docs_code="STAYED", status="Đình chỉ"),
        ]
    )

    selected = select_effective_documents(frame, as_of="2026-08-27")

    assert selected["docs_code"].tolist() == ["KEEP"]
    assert selected.iloc[0]["html_content"] == "<p>the longest retained version</p>"
    assert selected.iloc[0]["status"] == "Chưa có hiệu lực"


def test_build_excludes_unknown_effective_dates_and_records_the_reason(
    tmp_path, documents_with_missing_effective_date
):
    """Catches inferring an empty legal date or losing its auditable exclusion count."""
    from src.parser.corpus_builder import build_effective_corpus

    source = tmp_path / "raw.parquet"
    documents_with_missing_effective_date.to_parquet(source)

    manifest = build_effective_corpus(source, tmp_path / "output", as_of="2026-08-27")

    records = json.loads((tmp_path / "output" / "effective_legal_chunks.json").read_text(encoding="utf-8"))
    persisted_manifest = json.loads(
        (tmp_path / "output" / "effective_legal_corpus.manifest.json").read_text(encoding="utf-8")
    )
    assert {record["law_id"] for record in records} == {"INCLUDED"}
    assert manifest.excluded_missing_effective_date_count == 1
    assert manifest.excluded_missing_effective_date_reason == "unknown legal effective date"
    assert persisted_manifest["excluded_missing_effective_date_count"] == 1
    assert persisted_manifest["excluded_missing_effective_date_reason"] == "unknown legal effective date"


def test_build_manifest_reconciles_all_raw_document_selection_outcomes(tmp_path):
    """Catches a manifest that cannot reconcile raw rows to the final corpus."""
    from src.parser.corpus_builder import build_effective_corpus

    source = tmp_path / "raw.parquet"
    pd.DataFrame(
        [
            document(docs_code="KEEP"),
            document(docs_code="DUPLICATE", html_content="<p>short</p>"),
            document(docs_code="DUPLICATE", html_content="<p>longest retained duplicate</p>"),
            document(docs_code="UNKNOWN", effFrom=""),
            document(docs_code="FUTURE", effFrom="2026-09-01"),
            document(docs_code="EXPIRED", status="Hết hiệu lực"),
        ]
    ).to_parquet(source)

    manifest = build_effective_corpus(source, tmp_path / "output", as_of="2026-08-27")

    assert manifest.raw_document_count == 6
    assert manifest.eligible_document_count == 3
    assert manifest.excluded_missing_effective_date_count == 1
    assert manifest.excluded_future_effective_date_count == 1
    assert manifest.excluded_inactive_status_count == 1
    assert manifest.excluded_duplicate_document_count == 1
    assert manifest.document_count == 2
    assert (
        manifest.excluded_missing_effective_date_count
        + manifest.excluded_future_effective_date_count
        + manifest.excluded_inactive_status_count
        + manifest.eligible_document_count
        == manifest.raw_document_count
    )
    assert manifest.eligible_document_count == (
        manifest.excluded_duplicate_document_count + manifest.document_count
    )


def test_select_effective_documents_requires_exact_source_schema_and_dates():
    """Catches raw rows that silently bypass required metadata or use invalid effective dates."""
    from src.parser.corpus_builder import select_effective_documents

    missing_source_url = pd.DataFrame([document()]).drop(columns=["source_url"])
    with pytest.raises(CorpusValidationError, match="source_url"):
        select_effective_documents(missing_source_url, as_of=date(2026, 8, 27))

    invalid_date = pd.DataFrame([document(effFrom="2026/07/01")])
    with pytest.raises(CorpusValidationError, match="effFrom.*2026/07/01"):
        select_effective_documents(invalid_date, as_of="2026-08-27")


def test_year_only_effective_date_is_included_only_after_its_interval():
    """Catches treating an uncertain year as effective on an invented first day."""
    from src.parser.corpus_builder import select_effective_documents

    frame = pd.DataFrame([document(docs_code="YEAR", effFrom="2025")])

    assert select_effective_documents(frame, as_of="2024-12-31").empty
    assert select_effective_documents(frame, as_of="2026-01-01")["docs_code"].tolist() == ["YEAR"]


def test_year_only_effective_date_rejects_an_as_of_date_inside_its_interval():
    """Catches silently deciding eligibility when a supplied date falls within a year-only value."""
    from src.parser.corpus_builder import select_effective_documents

    with pytest.raises(CorpusValidationError, match="as_of falls inside year-only effFrom interval"):
        select_effective_documents(
            pd.DataFrame([document(effFrom="2025")]), as_of="2025-08-27"
        )


def test_year_only_effective_date_is_preserved_in_generated_chunks(tmp_path):
    """Catches normalization that loses the raw year-only legal effective-date value."""
    from src.parser.corpus_builder import build_effective_corpus

    source = tmp_path / "raw.parquet"
    pd.DataFrame([document(effFrom="2025")]).to_parquet(source)

    build_effective_corpus(source, tmp_path / "output", as_of="2026-08-27")

    records = json.loads((tmp_path / "output" / "effective_legal_chunks.json").read_text(encoding="utf-8"))
    assert {record["effective_date"] for record in records} == {"2025"}


@pytest.mark.parametrize("invalid_date", ["2026-7-1", "2026-07-1", "2026-7-01"])
def test_select_effective_documents_rejects_non_zero_padded_dates(invalid_date):
    """Catches date parsing that accepts non-canonical ISO date spellings."""
    from src.parser.corpus_builder import select_effective_documents

    with pytest.raises(CorpusValidationError, match=rf"effFrom.*{invalid_date}"):
        select_effective_documents(
            pd.DataFrame([document(effFrom=invalid_date)]), as_of="2026-08-27"
        )
    with pytest.raises(CorpusValidationError, match=rf"as_of.*{invalid_date}"):
        select_effective_documents(pd.DataFrame([document()]), as_of=invalid_date)


def test_article_chunker_emits_canonical_article_and_clause_metadata():
    """Catches chunking that loses citation metadata or merges separately numbered clauses."""
    from src.parser.corpus_builder import ArticleChunker

    chunks = ArticleChunker().chunk_document(pd.Series(document()))

    assert [chunk.chunk_id for chunk in chunks] == [
        "45-2026-QH_D1_K1",
        "45-2026-QH_D1_K2",
    ]
    assert chunks[0].law_id == "45/2026/QH"
    assert chunks[0].law_name == "Law on Corpus Fixtures"
    assert chunks[0].article_name == "Article 1. Scope"
    assert chunks[0].clause_name == "Clause 1"
    assert chunks[0].effective_date == "2026-07-01"
    assert chunks[0].status == "Chưa có hiệu lực"
    assert chunks[0].source_url == "https://example.test/law-45-2026"
    assert "A point in the first clause" in chunks[0].content
    assert chunks[1].clause_name == "Clause 2"


def test_article_chunker_falls_back_to_bounded_full_document_chunks(section_only_document):
    """Catches fabricated Article numbers or rejected effective documents without Article headings."""
    from src.parser.corpus_builder import ArticleChunker

    chunks = ArticleChunker().chunk_document(pd.Series(section_only_document))

    assert [chunk.chunk_id for chunk in chunks] == ["SECTION-ONLY_FULL_P1"]
    assert chunks[0].article_name == "Toàn văn"
    assert chunks[0].clause_name == ""
    assert "Nội dung toàn văn" in chunks[0].content
    assert 0 < len(chunks[0].content.split()) <= 390


def test_article_chunker_suffixes_repeated_clause_occurrences_deterministically(
    duplicate_clause_document,
):
    """Catches duplicate canonical IDs or occurrence suffixes that alter ordinary clauses."""
    from src.parser.corpus_builder import ArticleChunker

    chunks = ArticleChunker().chunk_document(pd.Series(duplicate_clause_document))

    assert [chunk.chunk_id for chunk in chunks] == [
        "DUPLICATE-CLAUSE_D4_K1",
        "DUPLICATE-CLAUSE_D4_K1_O2",
        "DUPLICATE-CLAUSE_D4_K2",
    ]
    assert [chunk.clause_name for chunk in chunks] == ["Clause 1", "Clause 1", "Clause 2"]
    assert "First top-level clause" in chunks[0].content
    assert "Embedded amended clause" in chunks[1].content


def test_article_chunker_suffixes_repeated_article_clause_occurrences(
    duplicate_article_document,
):
    """Catches duplicate IDs when a crawl repeats an entire canonical article."""
    from src.parser.corpus_builder import ArticleChunker

    chunks = ArticleChunker().chunk_document(pd.Series(duplicate_article_document))

    assert [chunk.chunk_id for chunk in chunks] == [
        "DUPLICATE-ARTICLE_D4_K1",
        "DUPLICATE-ARTICLE_D4_K1_O2",
    ]
    assert "First article occurrence" in chunks[0].content
    assert "Second article occurrence" in chunks[1].content


def test_article_chunker_stops_after_thirty_adjacent_numeric_table_lines():
    """Catches the counter reset that lets numeric table tails leak into legal article content."""
    from src.parser.corpus_builder import ArticleChunker

    numeric_lines = "".join("<p>1</p>" for _ in range(30))
    html = (
        "<article><p>Article 1. Table boundary</p><p>1. Retained clause.</p>"
        f"{numeric_lines}<p>Text after the table must not be retained.</p></article>"
    )

    chunks = ArticleChunker().chunk_document(pd.Series(document(html_content=html)))

    assert len(chunks) == 1
    assert "Retained clause" in chunks[0].content
    assert "Text after the table" not in chunks[0].content


def test_article_chunker_rejects_html_that_produces_no_chunkable_content():
    """Catches a fallback that silently writes a document without usable legal content."""
    from src.parser.corpus_builder import ArticleChunker

    with pytest.raises(CorpusValidationError, match="no chunkable content"):
        ArticleChunker().chunk_document(
            pd.Series(document(html_content="<article><p>   </p></article>"))
        )


def test_article_chunker_splits_long_content_with_bounded_overlap():
    """Catches oversize legal chunks or split boundaries that discard retrieval context."""
    from src.parser.corpus_builder import ArticleChunker

    words = " ".join(f"word{number:03d}" for number in range(730))
    html = f"<article><p>Article 1. Long provision</p><p>{words}</p></article>"

    chunks = ArticleChunker().chunk_document(pd.Series(document(html_content=html)))

    assert [chunk.chunk_id for chunk in chunks] == [
        "45-2026-QH_D1_K1_P1",
        "45-2026-QH_D1_K1_P2",
        "45-2026-QH_D1_K1_P3",
    ]
    assert all(0 < len(chunk.content.split()) <= 390 for chunk in chunks)
    assert chunks[0].content.split()[-40:] == chunks[1].content.split()[:40]
    assert chunks[1].content.split()[-40:] == chunks[2].content.split()[:40]


def test_article_chunker_prefers_complete_paragraph_boundaries_before_words():
    """Catches a packer that splits a new paragraph merely to fill remaining capacity."""
    from src.parser.corpus_builder import ArticleChunker

    first_paragraph = " ".join(f"paragraph_one_{number:03d}" for number in range(200))
    second_paragraph = " ".join(f"paragraph_two_{number:03d}" for number in range(200))
    html = (
        "<article><p>Article 1. Paragraph ordering</p>"
        f"<p>{first_paragraph}</p><p>{second_paragraph}</p></article>"
    )

    chunks = ArticleChunker().chunk_document(pd.Series(document(html_content=html)))

    assert len(chunks) == 2
    assert "paragraph_two_000" not in chunks[0].content
    assert chunks[0].content.split()[-1] == "paragraph_one_199"
    assert chunks[1].content.split()[40] == "paragraph_two_000"


def test_article_chunker_prefers_complete_sentence_boundaries_before_words():
    """Catches a packer that word-splits a sentence when its paragraph exceeds the limit."""
    from src.parser.corpus_builder import ArticleChunker

    first_sentence = " ".join(f"sentence_one_{number:03d}" for number in range(200)) + "."
    second_sentence = " ".join(f"sentence_two_{number:03d}" for number in range(200)) + "."
    html = (
        "<article><p>Article 1. Sentence ordering</p>"
        f"<p>{first_sentence} {second_sentence}</p></article>"
    )

    chunks = ArticleChunker().chunk_document(pd.Series(document(html_content=html)))

    assert len(chunks) == 2
    assert "sentence_two_000" not in chunks[0].content
    assert chunks[0].content.split()[-1] == "sentence_one_199."
    assert chunks[1].content.split()[40] == "sentence_two_000"


def test_build_effective_corpus_writes_hashed_json_and_manifest(tmp_path):
    """Catches a build that does not persist the selected corpus or binds its manifest to different bytes."""
    from src.parser.corpus_builder import build_effective_corpus

    source = tmp_path / "raw.parquet"
    pd.DataFrame([document(), document(docs_code="FUTURE", effFrom="2026-09-01")]).to_parquet(source)

    manifest = build_effective_corpus(source, tmp_path / "output", as_of="2026-08-27")

    chunk_path = tmp_path / "output" / "effective_legal_chunks.json"
    manifest_path = tmp_path / "output" / "effective_legal_corpus.manifest.json"
    chunk_bytes = chunk_path.read_bytes()
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [record["chunk_id"] for record in json.loads(chunk_bytes)] == ["45-2026-QH_D1_K1", "45-2026-QH_D1_K2"]
    assert manifest.document_count == 1
    assert manifest.chunk_count == 2
    assert manifest.as_of_date == "2026-08-27"
    assert manifest.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest.corpus_sha256 == hashlib.sha256(chunk_bytes).hexdigest()
    assert persisted_manifest == manifest.to_dict()
    assert not list((tmp_path / "output").glob("*.tmp"))


def test_build_records_full_document_fallback_counts(tmp_path, section_only_document):
    """Catches a corpus manifest that omits auditable non-Article fallback usage."""
    from src.parser.corpus_builder import build_effective_corpus

    source = tmp_path / "raw.parquet"
    pd.DataFrame([section_only_document]).to_parquet(source)

    manifest = build_effective_corpus(source, tmp_path / "output", as_of="2026-08-27")

    assert manifest.fallback_document_count == 1
    assert manifest.fallback_chunk_count == 1


def test_build_records_duplicate_clause_occurrence_count(tmp_path, duplicate_clause_document):
    """Catches a manifest that cannot disclose preserved repeated clause occurrences."""
    from src.parser.corpus_builder import build_effective_corpus

    source = tmp_path / "raw.parquet"
    pd.DataFrame([duplicate_clause_document]).to_parquet(source)

    manifest = build_effective_corpus(source, tmp_path / "output", as_of="2026-08-27")

    assert manifest.duplicate_clause_occurrence_count == 1


def test_build_suffixes_duplicate_normalized_document_clause_ids(tmp_path):
    """Catches normalized IDs colliding across raw document-code spellings."""
    from src.parser.corpus_builder import build_effective_corpus

    source = tmp_path / "raw.parquet"
    pd.DataFrame(
        [
            document(docs_code="98/2013/TT-BTC"),
            document(docs_code="98/2013/TT- BTC"),
        ]
    ).to_parquet(source)

    manifest = build_effective_corpus(source, tmp_path / "output", as_of="2026-08-27")

    records = json.loads(
        (tmp_path / "output" / "effective_legal_chunks.json").read_text(encoding="utf-8")
    )
    assert [record["chunk_id"] for record in records] == [
        "98-2013-TT-BTC_D1_K1",
        "98-2013-TT-BTC_D1_K2",
        "98-2013-TT-BTC_D1_K1_O2",
        "98-2013-TT-BTC_D1_K2_O2",
    ]
    assert manifest.duplicate_clause_occurrence_count == 2


def test_build_corpus_cli_reports_counts_and_rejects_invalid_dates(tmp_path):
    """Catches a CLI that hides build counts or exits successfully after invalid user input."""
    source = tmp_path / "raw.parquet"
    pd.DataFrame([document()]).to_parquet(source)
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "main_build_corpus.py"),
        "--source",
        str(source),
        "--output-dir",
        str(tmp_path / "output"),
        "--as-of",
        "2026-08-27",
    ]

    successful = subprocess.run(command, capture_output=True, text=True, check=False)
    invalid_date = subprocess.run(
        [*command[:-1], "2026/08/27"], capture_output=True, text=True, check=False
    )

    assert successful.returncode == 0
    assert "1 effective documents and 2 chunks" in successful.stdout
    assert invalid_date.returncode != 0
    assert "as_of must use YYYY-MM-DD" in invalid_date.stderr


def test_build_corpus_cli_reports_unknown_effective_date_exclusions(tmp_path):
    """Catches a successful build that hides excluded unknown legal dates from operators."""
    source = tmp_path / "raw.parquet"
    pd.DataFrame([document(), document(docs_code="UNKNOWN", effFrom="")]).to_parquet(source)

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "main_build_corpus.py"),
            "--source",
            str(source),
            "--output-dir",
            str(tmp_path / "output"),
            "--as-of",
            "2026-08-27",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Excluded 1 documents: unknown legal effective date." in completed.stdout


def test_build_corpus_cli_reports_full_document_fallback_counts(tmp_path, section_only_document):
    """Catches CLI output that hides Article-less fallback corpus coverage."""
    source = tmp_path / "raw.parquet"
    pd.DataFrame([section_only_document]).to_parquet(source)

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "main_build_corpus.py"),
            "--source",
            str(source),
            "--output-dir",
            str(tmp_path / "output"),
            "--as-of",
            "2026-08-27",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Fallback chunked 1 documents into 1 chunks without canonical Article headings." in completed.stdout


def test_build_corpus_cli_reports_duplicate_clause_occurrence_counts(
    tmp_path, duplicate_clause_document
):
    """Catches CLI output that hides preserved repeated legal clauses."""
    source = tmp_path / "raw.parquet"
    pd.DataFrame([duplicate_clause_document]).to_parquet(source)

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "main_build_corpus.py"),
            "--source",
            str(source),
            "--output-dir",
            str(tmp_path / "output"),
            "--as-of",
            "2026-08-27",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Recorded 1 duplicate clause occurrences with deterministic occurrence suffixes." in completed.stdout
