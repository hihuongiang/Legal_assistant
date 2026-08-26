import json

import pytest

from src.parser.chunk_builder import load_chunks
from src.parser.models import CorpusValidationError, LegalChunk


def canonical_record(**overrides: str) -> dict[str, str]:
    """A complete persisted chunk record, with literal citation metadata."""
    record = {
        "chunk_id": "A-01",
        "law_id": "45/2019/QH14",
        "law_name": "Law on Architecture",
        "article_name": "Article 12",
        "clause_name": "Clause 2",
        "effective_date": "2020-07-01",
        "status": "in_force",
        "source_url": "https://vanban.chinhphu.vn/law-45-2019",
        "content": "The retained legal text.",
    }
    record.update(overrides)
    return record


def test_legal_chunk_retains_citation_metadata_from_persisted_record():
    """Catches a loader/model that drops source, status, or legal citation fields."""
    chunk = LegalChunk.from_dict(canonical_record(), record_index=0)

    assert chunk.chunk_id == "A-01"
    assert chunk.law_id == "45/2019/QH14"
    assert chunk.law_name == "Law on Architecture"
    assert chunk.article_name == "Article 12"
    assert chunk.clause_name == "Clause 2"
    assert chunk.effective_date == "2020-07-01"
    assert chunk.status == "in_force"
    assert chunk.source_url == "https://vanban.chinhphu.vn/law-45-2019"
    assert chunk.content == "The retained legal text."


def test_legal_chunk_reports_record_index_and_missing_required_field():
    """Catches empty required content being accepted without an actionable error."""
    with pytest.raises(CorpusValidationError, match=r"record 7.*content"):
        LegalChunk.from_dict(canonical_record(content=""), record_index=7)


def test_load_chunks_rejects_duplicate_chunk_ids_with_both_row_indexes(tmp_path):
    """Catches duplicate IDs that would overwrite one another in downstream indexes."""
    path = tmp_path / "chunks.json"
    path.write_text(
        json.dumps([canonical_record(), canonical_record(content="Repeated ID")]),
        encoding="utf-8",
    )

    with pytest.raises(CorpusValidationError, match=r"duplicate chunk_id 'A-01'.*0.*1"):
        load_chunks(path)
