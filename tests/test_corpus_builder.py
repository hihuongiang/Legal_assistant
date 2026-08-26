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


def test_select_effective_documents_requires_exact_source_schema_and_dates():
    """Catches raw rows that silently bypass required metadata or use ambiguous effective dates."""
    from src.parser.corpus_builder import select_effective_documents

    missing_source_url = pd.DataFrame([document()]).drop(columns=["source_url"])
    with pytest.raises(CorpusValidationError, match="source_url"):
        select_effective_documents(missing_source_url, as_of=date(2026, 8, 27))

    invalid_date = pd.DataFrame([document(effFrom="2026/07/01")])
    with pytest.raises(CorpusValidationError, match="effFrom.*2026/07/01"):
        select_effective_documents(invalid_date, as_of="2026-08-27")


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


def test_article_chunker_rejects_html_that_produces_no_legal_chunks():
    """Catches a parser that silently writes a document without usable legal content."""
    from src.parser.corpus_builder import ArticleChunker

    with pytest.raises(CorpusValidationError, match="no article chunks"):
        ArticleChunker().chunk_document(
            pd.Series(document(html_content="<article><p>Unstructured crawl text only.</p></article>"))
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
