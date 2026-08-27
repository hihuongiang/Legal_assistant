"""Build a FAISS index bound to the canonical processed legal corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from src.parser.chunk_builder import load_chunks
from src.parser.models import CorpusValidationError
from src.retrieval.faiss_store import FaissStore
from src.retrieval.index_manifest import IndexValidationError


PROCESSED_DIR = Path("data/processed")
DEFAULT_CHUNKS_PATH = PROCESSED_DIR / "effective_legal_chunks.json"
DEFAULT_INDEX_PATH = PROCESSED_DIR / "effective_legal_chunks.faiss"
DEFAULT_MANIFEST_PATH = PROCESSED_DIR / "effective_legal_chunks.manifest.json"
DEFAULT_CORPUS_MANIFEST_PATH = PROCESSED_DIR / "effective_legal_corpus.manifest.json"
DenseEmbedder = None


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _validate_processed_paths(paths: list[Path], allow_non_processed: bool) -> None:
    if allow_non_processed:
        return
    outside = [str(path) for path in paths if not _is_within(path, PROCESSED_DIR)]
    if outside:
        raise CorpusValidationError(
            "index inputs and outputs must be under data/processed; "
            "pass --allow-non-processed to override: " + ", ".join(outside)
        )


def _read_corpus_as_of(path: Path) -> str:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusValidationError("corpus build manifest must be valid UTF-8 JSON") from error
    as_of_date = manifest.get("as_of_date") if isinstance(manifest, dict) else None
    if not isinstance(as_of_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of_date):
        raise CorpusValidationError("corpus build manifest must include as_of_date in YYYY-MM-DD format")
    return as_of_date


def build_index(
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    corpus_manifest_path: str | Path = DEFAULT_CORPUS_MANIFEST_PATH,
    *,
    allow_non_processed: bool = False,
) -> int:
    """Build and persist an index and its corpus-bound JSON manifest."""
    paths = [Path(chunks_path), Path(index_path), Path(manifest_path), Path(corpus_manifest_path)]
    _validate_processed_paths(paths, allow_non_processed)
    chunks_path, index_path, manifest_path, corpus_manifest_path = paths
    as_of_date = _read_corpus_as_of(corpus_manifest_path)
    chunks = load_chunks(chunks_path)

    embedder_factory = DenseEmbedder
    if embedder_factory is None:
        from src.embedding.dense_model import DenseEmbedder as embedder_factory

    embedder = embedder_factory()
    store = FaissStore(embedder)
    store.build_index(chunks)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    store.save(index_path, manifest_path, chunks, as_of_date=as_of_date)
    print(f"Built {len(chunks)} FAISS vectors as of {as_of_date}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a FAISS index from the canonical processed corpus.")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--corpus-manifest", default=str(DEFAULT_CORPUS_MANIFEST_PATH))
    parser.add_argument("--allow-non-processed", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        return build_index(
            chunks_path=arguments.chunks,
            index_path=arguments.index,
            manifest_path=arguments.manifest,
            corpus_manifest_path=arguments.corpus_manifest,
            allow_non_processed=arguments.allow_non_processed,
        )
    except (CorpusValidationError, IndexValidationError, FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    sys.exit(main())
