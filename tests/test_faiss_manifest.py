import hashlib
import json
import pickle
from dataclasses import replace
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from src.parser.models import LegalChunk


def legal_chunk(chunk_id: str, content: str) -> LegalChunk:
    return LegalChunk(
        chunk_id=chunk_id,
        law_id="LAW-1",
        law_name="Fixture law",
        article_name="Article 1",
        clause_name="Clause 1",
        effective_date="2026-01-01",
        status="Effective",
        source_url="https://example.test/law-1",
        content=content,
    )


def test_index_manifest_serializes_a_canonical_ordered_corpus_fingerprint(tmp_path):
    """Catches a manifest that omits the fields binding an index to this exact corpus."""
    from src.retrieval.index_manifest import IndexManifest

    chunks = [legal_chunk("LAW-1-A", "first provision"), legal_chunk("LAW-1-B", "second provision")]
    manifest = IndexManifest.create(
        chunks=chunks,
        as_of_date="2026-08-27",
        vector_count=2,
        embedding_dimension=3,
        model_name="fixture-embedder-v1",
    )
    manifest_path = tmp_path / "effective_legal_chunks.manifest.json"
    manifest.write(manifest_path)

    expected_payload = '[{"chunk_id":"LAW-1-A","content":"first provision"},{"chunk_id":"LAW-1-B","content":"second provision"}]'
    assert manifest.schema_version == 1
    assert manifest.ordered_chunk_ids == ("LAW-1-A", "LAW-1-B")
    assert manifest.corpus_sha256 == hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "ordered_chunk_ids": ["LAW-1-A", "LAW-1-B"],
        "corpus_sha256": hashlib.sha256(expected_payload.encode("utf-8")).hexdigest(),
        "as_of_date": "2026-08-27",
        "vector_count": 2,
        "embedding_dimension": 3,
        "model_name": "fixture-embedder-v1",
    }


class FakeEmbeddingModel:
    def __init__(self, dimension: int):
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


class FakeEmbedder:
    def __init__(self, dimension: int = 3, model_name: str = "fixture-embedder-v1"):
        self.model = FakeEmbeddingModel(dimension)
        self.model_name = model_name

    def encode_chunks(self, texts: list[str]) -> np.ndarray:
        vectors = {
            "first provision": [1.0, 0.0, 0.0],
            "second provision": [0.0, 1.0, 0.0],
        }
        return np.asarray([vectors[text][: self.model.dimension] for text in texts], dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_chunks([query])[0]


def fixture_chunks() -> list[LegalChunk]:
    return [legal_chunk("LAW-1-A", "first provision"), legal_chunk("LAW-1-B", "second provision")]


def test_faiss_store_loads_a_manifest_bound_index_and_searches(tmp_path):
    """Catches a load path that accepts an index without proving it matches the active corpus."""
    from src.retrieval.faiss_store import FaissStore

    chunks = fixture_chunks()
    index_path = tmp_path / "effective_legal_chunks.faiss"
    manifest_path = tmp_path / "effective_legal_chunks.manifest.json"
    built_store = FaissStore(FakeEmbedder())
    built_store.build_index(chunks)
    built_store.save(index_path, manifest_path, chunks, as_of_date="2026-08-27")

    loaded_store = FaissStore(FakeEmbedder())
    loaded_store.load(index_path, manifest_path, chunks)

    assert loaded_store.search("first provision", top_k=1) == [("LAW-1-A", 1.0)]


def built_index_paths(tmp_path):
    from src.retrieval.faiss_store import FaissStore

    chunks = fixture_chunks()
    index_path = tmp_path / "effective_legal_chunks.faiss"
    manifest_path = tmp_path / "effective_legal_chunks.manifest.json"
    store = FaissStore(FakeEmbedder())
    store.build_index(chunks)
    store.save(index_path, manifest_path, chunks, as_of_date="2026-08-27")
    return index_path, manifest_path, chunks


def test_faiss_store_rejects_a_different_chunk_corpus_before_search(tmp_path):
    """Catches a load that maps vectors to newly introduced corpus IDs."""
    from src.retrieval.faiss_store import FaissStore
    from src.retrieval.index_manifest import IndexValidationError

    index_path, manifest_path, chunks = built_index_paths(tmp_path)
    changed_corpus = [legal_chunk("LAW-1-C", "first provision"), chunks[1]]

    with pytest.raises(IndexValidationError, match="ID order"):
        FaissStore(FakeEmbedder()).load(index_path, manifest_path, changed_corpus)


def test_faiss_store_rejects_changed_chunk_content_before_search(tmp_path):
    """Catches a load that reuses vectors after canonical chunk text changes."""
    from src.retrieval.faiss_store import FaissStore
    from src.retrieval.index_manifest import IndexValidationError

    index_path, manifest_path, chunks = built_index_paths(tmp_path)
    changed_content = [replace(chunks[0], content="amended provision"), chunks[1]]

    with pytest.raises(IndexValidationError, match="corpus hash"):
        FaissStore(FakeEmbedder()).load(index_path, manifest_path, changed_content)


def test_faiss_store_rejects_reordered_chunk_ids_before_search(tmp_path):
    """Catches a load that silently maps a vector position to the wrong chunk ID."""
    from src.retrieval.faiss_store import FaissStore
    from src.retrieval.index_manifest import IndexValidationError

    index_path, manifest_path, chunks = built_index_paths(tmp_path)

    with pytest.raises(IndexValidationError, match="ID order"):
        FaissStore(FakeEmbedder()).load(index_path, manifest_path, list(reversed(chunks)))


def test_faiss_store_rejects_a_corpus_with_the_wrong_vector_count_before_search(tmp_path):
    """Catches a load that accepts fewer chunks than vectors in the persisted index."""
    from src.retrieval.faiss_store import FaissStore
    from src.retrieval.index_manifest import IndexValidationError

    index_path, manifest_path, chunks = built_index_paths(tmp_path)

    with pytest.raises(IndexValidationError, match="vector count"):
        FaissStore(FakeEmbedder()).load(index_path, manifest_path, chunks[:1])


def test_faiss_store_rejects_a_different_embedding_dimension_before_search(tmp_path):
    """Catches a load that queries vectors with an incompatible embedding dimension."""
    from src.retrieval.faiss_store import FaissStore
    from src.retrieval.index_manifest import IndexValidationError

    index_path, manifest_path, chunks = built_index_paths(tmp_path)

    with pytest.raises(IndexValidationError, match="dimension"):
        FaissStore(FakeEmbedder(dimension=2)).load(index_path, manifest_path, chunks)


def test_faiss_store_rejects_a_different_embedding_model_before_search(tmp_path):
    """Catches a load that queries an index created by a different embedding model."""
    from src.retrieval.faiss_store import FaissStore
    from src.retrieval.index_manifest import IndexValidationError

    index_path, manifest_path, chunks = built_index_paths(tmp_path)

    with pytest.raises(IndexValidationError, match="model"):
        FaissStore(FakeEmbedder(model_name="fixture-embedder-v2")).load(
            index_path, manifest_path, chunks
        )


def test_faiss_store_rejects_duplicate_chunk_ids_before_search(tmp_path):
    """Catches a load that accepts an ambiguous corpus-to-vector mapping."""
    from src.retrieval.faiss_store import FaissStore
    from src.retrieval.index_manifest import IndexValidationError

    index_path, manifest_path, chunks = built_index_paths(tmp_path)
    duplicate_ids = [chunks[0], replace(chunks[1], chunk_id=chunks[0].chunk_id)]

    with pytest.raises(IndexValidationError, match="duplicate"):
        FaissStore(FakeEmbedder()).load(index_path, manifest_path, duplicate_ids)


def test_faiss_store_does_not_silently_load_legacy_pickle_metadata(tmp_path):
    """Catches a fallback that would deserialize unsafe legacy pickle metadata."""
    from src.retrieval.faiss_store import FaissStore
    from src.retrieval.index_manifest import IndexValidationError

    index_path, manifest_path, chunks = built_index_paths(tmp_path)
    manifest_path.write_bytes(pickle.dumps([chunk.chunk_id for chunk in chunks]))

    with pytest.raises(IndexValidationError, match="JSON"):
        FaissStore(FakeEmbedder()).load(index_path, manifest_path, chunks)


def write_processed_corpus(root: Path, chunks: list[LegalChunk]) -> Path:
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True)
    chunks_path = processed_dir / "effective_legal_chunks.json"
    chunks_path.write_text(json.dumps([asdict(chunk) for chunk in chunks]), encoding="utf-8")
    (processed_dir / "effective_legal_corpus.manifest.json").write_text(
        json.dumps({"as_of_date": "2026-08-27"}), encoding="utf-8"
    )
    return chunks_path


def test_build_index_uses_processed_defaults_and_corpus_build_as_of(tmp_path, monkeypatch):
    """Catches a build command that falls back to legacy artifacts or invents its own as-of date."""
    import main_build_index
    from src.retrieval.index_manifest import IndexManifest

    write_processed_corpus(tmp_path, fixture_chunks())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_build_index, "DenseEmbedder", FakeEmbedder)

    assert main_build_index.main([]) == 0

    manifest = IndexManifest.read(tmp_path / "data" / "processed" / "effective_legal_chunks.manifest.json")
    assert manifest.as_of_date == "2026-08-27"
    assert (tmp_path / "data" / "processed" / "effective_legal_chunks.faiss").is_file()


def test_build_index_rejects_non_processed_paths_without_an_explicit_override(capsys):
    """Catches a CLI that can accidentally build an index from unvalidated raw or legacy chunks."""
    import main_build_index

    with pytest.raises(SystemExit) as error:
        main_build_index.main(["--chunks", "data/legacy/legal_chunks.json"])

    assert error.value.code == 2
    assert "data/processed" in capsys.readouterr().err


def test_build_index_allows_non_processed_paths_only_with_the_override(tmp_path, monkeypatch):
    """Catches an override flag that is accepted by argparse but ignored by the builder."""
    import main_build_index
    from src.retrieval.index_manifest import IndexManifest

    corpus_dir = tmp_path / "isolated"
    corpus_dir.mkdir()
    chunks_path = corpus_dir / "chunks.json"
    chunks_path.write_text(json.dumps([asdict(chunk) for chunk in fixture_chunks()]), encoding="utf-8")
    corpus_manifest_path = corpus_dir / "corpus.manifest.json"
    corpus_manifest_path.write_text(json.dumps({"as_of_date": "2026-08-27"}), encoding="utf-8")
    index_path = corpus_dir / "chunks.faiss"
    manifest_path = corpus_dir / "chunks.manifest.json"
    monkeypatch.setattr(main_build_index, "DenseEmbedder", FakeEmbedder)

    assert main_build_index.main(
        [
            "--chunks", str(chunks_path),
            "--index", str(index_path),
            "--manifest", str(manifest_path),
            "--corpus-manifest", str(corpus_manifest_path),
            "--allow-non-processed",
        ]
    ) == 0

    assert IndexManifest.read(manifest_path).as_of_date == "2026-08-27"
