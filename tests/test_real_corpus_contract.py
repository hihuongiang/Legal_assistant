"""Contracts that validate a locally generated production corpus when present."""

from pathlib import Path

import pytest

from src.parser.chunk_builder import load_chunks


PROCESSED_CHUNKS_PATH = Path("data/processed/effective_legal_chunks.json")


@pytest.mark.skipif(
    not PROCESSED_CHUNKS_PATH.exists(),
    reason="build corpus first",
)
def test_generated_chunks_are_unique_and_bounded():
    """Reject generated artifacts with duplicate IDs or oversized content."""
    chunks = load_chunks(PROCESSED_CHUNKS_PATH)

    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(len(chunk.content.split()) <= 390 for chunk in chunks)
