import torch
from sentence_transformers import SentenceTransformer
import numpy as np


class DenseEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None
    ):
        """
        Load BGE-M3 model.
        """

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            model_name,
            device=device
        )

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Input:
            texts: list văn bản

        Output:
            ndarray shape = (n_samples, embedding_dim)
        """

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode một câu hỏi.
        """
        return self.encode([query])[0]

    def encode_chunks(self, chunks: list[str]) -> np.ndarray:
        """
        Encode nhiều chunk luật.
        """
        return self.encode(chunks)