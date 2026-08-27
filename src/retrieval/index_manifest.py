"""Versioned metadata that binds a FAISS index to one canonical chunk corpus."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

from src.parser.models import LegalChunk


INDEX_MANIFEST_SCHEMA_VERSION = 1


class IndexValidationError(ValueError):
    """Raised when an index cannot be safely matched with a chunk corpus."""


def corpus_fingerprint(chunks: Sequence[LegalChunk]) -> str:
    """Hash ordered retrieval identity and text using canonical compact JSON."""
    payload = [
        {"chunk_id": chunk.chunk_id, "content": chunk.content}
        for chunk in chunks
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class IndexManifest:
    schema_version: int
    ordered_chunk_ids: tuple[str, ...]
    corpus_sha256: str
    as_of_date: str
    vector_count: int
    embedding_dimension: int
    model_name: str

    @classmethod
    def create(
        cls,
        *,
        chunks: Sequence[LegalChunk],
        as_of_date: str,
        vector_count: int,
        embedding_dimension: int,
        model_name: str,
    ) -> "IndexManifest":
        return cls(
            schema_version=INDEX_MANIFEST_SCHEMA_VERSION,
            ordered_chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
            corpus_sha256=corpus_fingerprint(chunks),
            as_of_date=as_of_date,
            vector_count=vector_count,
            embedding_dimension=embedding_dimension,
            model_name=model_name,
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["ordered_chunk_ids"] = list(self.ordered_chunk_ids)
        return data

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: str | Path) -> "IndexManifest":
        """Read and validate a JSON manifest; pickle metadata is intentionally unsupported."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise IndexValidationError("index manifest must be valid UTF-8 JSON") from error
        if not isinstance(data, dict):
            raise IndexValidationError("index manifest must be a JSON object")

        required = {
            "schema_version",
            "ordered_chunk_ids",
            "corpus_sha256",
            "as_of_date",
            "vector_count",
            "embedding_dimension",
            "model_name",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise IndexValidationError(f"index manifest is missing required field(s): {', '.join(missing)}")
        if data["schema_version"] != INDEX_MANIFEST_SCHEMA_VERSION:
            raise IndexValidationError("unsupported index manifest schema version")
        chunk_ids = data["ordered_chunk_ids"]
        if not isinstance(chunk_ids, list) or not all(
            isinstance(chunk_id, str) and chunk_id for chunk_id in chunk_ids
        ):
            raise IndexValidationError("ordered_chunk_ids must be a list of non-empty strings")
        if not isinstance(data["corpus_sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", data["corpus_sha256"]
        ):
            raise IndexValidationError("corpus_sha256 must be a SHA-256 hex digest")
        if not isinstance(data["as_of_date"], str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", data["as_of_date"]
        ):
            raise IndexValidationError("as_of_date must use YYYY-MM-DD")
        for field in ("vector_count", "embedding_dimension"):
            if not isinstance(data[field], int) or isinstance(data[field], bool) or data[field] < 0:
                raise IndexValidationError(f"{field} must be a non-negative integer")
        if not isinstance(data["model_name"], str) or not data["model_name"].strip():
            raise IndexValidationError("model_name must be a non-empty string")

        return cls(
            schema_version=data["schema_version"],
            ordered_chunk_ids=tuple(chunk_ids),
            corpus_sha256=data["corpus_sha256"],
            as_of_date=data["as_of_date"],
            vector_count=data["vector_count"],
            embedding_dimension=data["embedding_dimension"],
            model_name=data["model_name"],
        )
