"""Offline behavior tests for canonical submission formatting."""

import json

import pytest

from src.parser.models import LegalChunk


def legal_chunk(
    chunk_id: str,
    *,
    law_id: str = "59/2020/QH14",
    law_name: str = "Law on Enterprises",
    article_name: str = "Article 1",
    clause_name: str = "",
) -> LegalChunk:
    return LegalChunk(
        chunk_id=chunk_id,
        law_id=law_id,
        law_name=law_name,
        article_name=article_name,
        clause_name=clause_name,
        effective_date="2020-01-01",
        status="In force",
        source_url="https://example.test/law",
        content="Canonical content",
    )


def test_format_submission_item_uses_stable_deduplicated_chunk_metadata_only():
    """Catches citations that depend on chunk IDs, prompt text, or set ordering."""
    from src.submission.formatter import format_submission_item

    chunks = {
        "first": legal_chunk("first"),
        "second": legal_chunk("second", article_name="Article 2"),
        "other": legal_chunk(
            "other", law_id="12/2021/ND", law_name="Decree on Records", article_name="Article 3"
        ),
    }

    formatted = format_submission_item(
        {"id": "q-1", "question": "What applies?"},
        {"answer": "The answer", "relevant_articles": ["second", "first", "second", "missing", "other"]},
        chunks,
    )

    assert formatted == {
        "id": "q-1",
        "question": "What applies?",
        "answer": "The answer",
        "relevant_docs": [
            "59/2020/QH14|Law on Enterprises",
            "12/2021/ND|Decree on Records",
        ],
        "relevant_articles": [
            "59/2020/QH14|Law on Enterprises|Article 2",
            "59/2020/QH14|Law on Enterprises|Article 1",
            "12/2021/ND|Decree on Records|Article 3",
        ],
    }


def test_write_submission_materializes_all_items_before_replacing_existing_target(tmp_path):
    """Catches a failed submission run that truncates or replaces the prior results file."""
    from src.submission.formatter import write_submission

    output = tmp_path / "results.json"
    output.write_text("previous complete results", encoding="utf-8")

    def failing_items():
        yield {"id": "q-1"}
        raise RuntimeError("pipeline failed")

    with pytest.raises(RuntimeError, match="pipeline failed"):
        write_submission(output, failing_items())

    assert output.read_text(encoding="utf-8") == "previous complete results"
    assert list(tmp_path.glob(".results.json.*.tmp")) == []


def test_write_submission_replaces_target_with_valid_utf8_json(tmp_path):
    """Catches an output write that omits non-ASCII answer text or invalid JSON."""
    from src.submission.formatter import write_submission

    output = tmp_path / "results.json"
    write_submission(output, [{"answer": "Điều luật áp dụng"}])

    assert json.loads(output.read_text(encoding="utf-8")) == [{"answer": "Điều luật áp dụng"}]
