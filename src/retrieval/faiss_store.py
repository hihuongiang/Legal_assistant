import faiss
import numpy as np
from tqdm import tqdm
from typing import Protocol

from src.parser.chunk_builder import LegalChunk
from src.retrieval.index_manifest import IndexManifest, IndexValidationError, corpus_fingerprint


class EmbeddingModel(Protocol):
    def get_sentence_embedding_dimension(self) -> int: ...


class Embedder(Protocol):
    model: EmbeddingModel
    model_name: str

    def encode_chunks(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, query: str) -> np.ndarray: ...


class FaissStore:

    def __init__(self, embedder: Embedder):

        self.embedder = embedder

        self.embedding_dim = (
            self.embedder.model.get_sentence_embedding_dimension()
        )
        self.model_name = self._embedder_model_name()

        self.index = self._new_index()

        self.chunk_ids = []

    def _new_index(self):
        return faiss.IndexFlatIP(self.embedding_dim)

    def _embedder_model_name(self) -> str:
        model_name = getattr(self.embedder, "model_name", None)
        if not model_name:
            model_name = getattr(self.embedder.model, "model_name_or_path", None)
        if not isinstance(model_name, str) or not model_name.strip():
            raise IndexValidationError("embedder must expose a non-empty model_name")
        return model_name.strip()

    @staticmethod
    def _validate_unique_chunk_ids(chunk_ids: list[str], source: str) -> None:
        if len(chunk_ids) != len(set(chunk_ids)):
            raise IndexValidationError(f"duplicate chunk IDs in {source}")

    def build_index(
        self,
        chunks: list[LegalChunk],
        batch_size: int = 32
    ) -> None:
        """
        Build FAISS index theo từng batch để tránh OOM.
        """

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        self._validate_unique_chunk_ids(chunk_ids, "chunk corpus")
        self.index = self._new_index()
        self.chunk_ids = []

        for i in tqdm(
            range(0, len(chunks), batch_size),
            desc="Building FAISS"
        ):

            batch = chunks[i:i + batch_size]

            texts = [
                chunk.content
                for chunk in batch
            ]

            embeddings = self.embedder.encode_chunks(
                texts
            )

            embeddings = embeddings.astype(
                np.float32
            )

            self.index.add(
                embeddings
            )

            self.chunk_ids.extend(
                chunk.chunk_id
                for chunk in batch
            )

    def search(
        self,
        query: str,
        top_k: int = 30
    ) -> list[tuple[str, float]]:

        query_vector = self.embedder.encode_query(query)

        query_vector = query_vector.astype(
            np.float32
        )

        query_vector = np.expand_dims(
            query_vector,
            axis=0
        )

        scores, indices = self.index.search(
            query_vector,
            top_k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx == -1:
                continue

            results.append(
                (
                    self.chunk_ids[idx],
                    float(score)
                )
            )

        return results

    def save(
        self,
        index_path: str,
        manifest_path: str,
        chunks: list[LegalChunk],
        *,
        as_of_date: str,
    ) -> None:
        """Persist the index with JSON metadata bound to the canonical chunks."""
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        self._validate_unique_chunk_ids(chunk_ids, "chunk corpus")
        if chunk_ids != self.chunk_ids:
            raise IndexValidationError("cannot save an index built from a different chunk ID order")
        if self.index.ntotal != len(chunks):
            raise IndexValidationError("cannot save an index whose vector count differs from the corpus")

        manifest = IndexManifest.create(
            chunks=chunks,
            as_of_date=as_of_date,
            vector_count=self.index.ntotal,
            embedding_dimension=self.embedding_dim,
            model_name=self.model_name,
        )
        faiss.write_index(self.index, str(index_path))
        manifest.write(manifest_path)

    def load(
        self,
        index_path: str,
        manifest_path: str,
        chunks: list[LegalChunk],
    ) -> None:
        """Load only an index whose manifest proves it matches ``chunks`` and this embedder."""
        manifest = IndexManifest.read(manifest_path)
        loaded_index = faiss.read_index(str(index_path))
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        self._validate_unique_chunk_ids(chunk_ids, "chunk corpus")
        self._validate_unique_chunk_ids(list(manifest.ordered_chunk_ids), "index manifest")

        if manifest.vector_count != loaded_index.ntotal or manifest.vector_count != len(chunk_ids):
            raise IndexValidationError("index vector count does not match the manifest and corpus")
        if (
            manifest.embedding_dimension != loaded_index.d
            or manifest.embedding_dimension != self.embedding_dim
        ):
            raise IndexValidationError("index embedding dimension does not match the active embedder")
        if manifest.model_name != self.model_name:
            raise IndexValidationError("index embedding model does not match the active embedder")
        if manifest.ordered_chunk_ids != tuple(chunk_ids):
            raise IndexValidationError("index chunk ID order does not match the active corpus")
        if manifest.corpus_sha256 != corpus_fingerprint(chunks):
            raise IndexValidationError("index corpus hash does not match the active corpus")

        self.index = loaded_index
        self.chunk_ids = list(manifest.ordered_chunk_ids)
