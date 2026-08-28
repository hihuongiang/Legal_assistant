from rank_bm25 import BM25Okapi
from pyvi.ViTokenizer import tokenize

from src.parser.chunk_builder import LegalChunk


class BM25Retriever:
    def __init__(self):
        """
        BM25 retriever.
        """
        self.bm25 = None
        self.chunk_ids = []

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize tiếng Việt bằng PyVi.
        """

        text = tokenize(text)

        return text.split()

    def build_index(
            self,
            chunks: list[LegalChunk]
    ) -> None:
        """
        Xây dựng BM25 index từ list chunk.
        """

        corpus = []

        for chunk in chunks:

            tokens = self._tokenize(chunk.content)

            corpus.append(tokens)

        self.bm25 = BM25Okapi(corpus)

        self.chunk_ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

    def search(
            self,
            query: str,
            top_k: int = 30
    ) -> list[tuple[str, float]]:
        """
        Trả về:
        [
            (chunk_id, score),
            ...
        ]
        """

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        indexed_scores = list(
            zip(
                self.chunk_ids,
                scores
            )
        )

        indexed_scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return indexed_scores[:top_k]