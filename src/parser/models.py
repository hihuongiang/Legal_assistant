"""Canonical, validated data models for the legal corpus."""

from dataclasses import asdict, dataclass


class CorpusValidationError(ValueError):
    """Raised when persisted corpus data cannot satisfy the canonical contract."""


@dataclass(frozen=True)
class LegalChunk:
    chunk_id: str
    law_id: str
    law_name: str
    article_name: str
    clause_name: str
    effective_date: str
    status: str
    source_url: str
    content: str

    @classmethod
    def from_dict(cls, record: dict, record_index: int) -> "LegalChunk":
        """Build a validated chunk from one persisted JSON record."""
        if not isinstance(record, dict):
            raise CorpusValidationError(f"record {record_index} must be an object")

        required_fields = (
            "chunk_id",
            "law_id",
            "law_name",
            "article_name",
            "effective_date",
            "status",
            "source_url",
            "content",
        )
        values: dict[str, str] = {}
        for field in required_fields:
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise CorpusValidationError(
                    f"record {record_index} is missing required field '{field}'"
                )
            values[field] = value.strip()

        clause_name = record.get("clause_name", "")
        if clause_name is None:
            clause_name = ""
        if not isinstance(clause_name, str):
            raise CorpusValidationError(
                f"record {record_index} has invalid field 'clause_name'"
            )

        return cls(clause_name=clause_name.strip(), **values)


@dataclass(frozen=True)
class CorpusBuildManifest:
    """Checksums and counts that identify a deterministic corpus build."""

    as_of_date: str
    source_sha256: str
    document_count: int
    chunk_count: int
    corpus_sha256: str

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-serializable representation."""
        return asdict(self)
