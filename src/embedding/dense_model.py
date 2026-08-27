"""CUDA-only, memory-bounded dense embedding."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.runtime.gpu import GpuMemoryError, release_cuda_model, require_cuda


CUDA_BATCH_SIZE = 1
CUDA_SEQUENCE_LIMIT = 512


class DenseEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
    ):
        """Load the dense model exclusively on CUDA in FP16."""
        require_cuda()
        if device not in (None, "cuda"):
            raise ValueError("DenseEmbedder supports CUDA execution only.")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cuda")
        self.model.max_seq_length = CUDA_SEQUENCE_LIMIT
        self.model.half()

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
        batch_size: int = CUDA_BATCH_SIZE,
    ) -> np.ndarray:
        """Encode text under the fixed safe CUDA batch and sequence limits."""
        del batch_size  # Callers cannot raise the fixed GPU memory bound.
        if self.model is None:
            raise RuntimeError("DenseEmbedder is closed.")

        try:
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                return self.model.encode(
                    texts,
                    batch_size=CUDA_BATCH_SIZE,
                    normalize_embeddings=normalize_embeddings,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
        except torch.OutOfMemoryError as error:
            raise GpuMemoryError("embed", CUDA_BATCH_SIZE, CUDA_SEQUENCE_LIMIT) from error

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query."""
        return self.encode([query])[0]

    def encode_chunks(
        self,
        chunks: list[str],
        batch_size: int = CUDA_BATCH_SIZE,
    ) -> np.ndarray:
        """Encode corpus chunks with the fixed safe CUDA batch size."""
        return self.encode(chunks, batch_size=batch_size)

    def close(self) -> None:
        """Release the CUDA model; calling close repeatedly is safe."""
        release_cuda_model(self)
