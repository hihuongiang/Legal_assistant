"""CUDA-only, memory-bounded BGE reranking."""

import torch
from sentence_transformers import CrossEncoder

from src.parser.chunk_builder import LegalChunk
from src.runtime.gpu import GpuMemoryError, release_cuda_model, require_cuda


CUDA_BATCH_SIZE = 1
CUDA_SEQUENCE_LIMIT = 512


class BGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
    ):
        """Load the cross encoder exclusively on CUDA in FP16."""
        del use_fp16  # CUDA FP16 is mandatory for bounded model execution.
        require_cuda()
        self.model = CrossEncoder(
            model_name,
            max_length=CUDA_SEQUENCE_LIMIT,
            device="cuda",
        )
        self.model.model.half()

    def rerank(
        self,
        query: str,
        chunks: list[LegalChunk],
        top_k: int = 5,
    ) -> list[tuple[LegalChunk, float]]:
        """Return the top-ranked chunks under a fixed CUDA prediction batch size."""
        if not chunks:
            return []
        if self.model is None:
            raise RuntimeError("BGEReranker is closed.")

        pairs = [(query, chunk.content) for chunk in chunks]
        try:
            scores = self.model.predict(pairs, batch_size=CUDA_BATCH_SIZE)
        except torch.OutOfMemoryError as error:
            raise GpuMemoryError("rerank", CUDA_BATCH_SIZE, CUDA_SEQUENCE_LIMIT) from error

        results = list(zip(chunks, scores))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    def close(self) -> None:
        """Release the CUDA model; calling close repeatedly is safe."""
        release_cuda_model(self)
