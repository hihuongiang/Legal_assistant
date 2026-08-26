"""Loading helpers for the canonical legal chunk representation."""

import json
from pathlib import Path

from src.parser.models import CorpusValidationError, LegalChunk


def load_chunks(path: str | Path) -> list[LegalChunk]:
    """Load a canonical JSON chunk list and validate its IDs and metadata."""
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            records = json.load(file)
    except json.JSONDecodeError as error:
        raise CorpusValidationError("chunk corpus must contain valid JSON") from error

    if not isinstance(records, list):
        raise CorpusValidationError("chunk corpus must be a JSON list")

    chunks: list[LegalChunk] = []
    indexes_by_id: dict[str, int] = {}
    for record_index, record in enumerate(records):
        chunk = LegalChunk.from_dict(record, record_index)
        first_index = indexes_by_id.get(chunk.chunk_id)
        if first_index is not None:
            raise CorpusValidationError(
                f"duplicate chunk_id '{chunk.chunk_id}' at records {first_index} and {record_index}"
            )
        indexes_by_id[chunk.chunk_id] = record_index
        chunks.append(chunk)

    return chunks
