from collections import defaultdict


class RRF:
    def __init__(self, k: int = 60):
        """
        Reciprocal Rank Fusion

        score = Σ 1 / (k + rank)

        k=60 là giá trị thường dùng.
        """
        self.k = k

    def fuse(
        self,
        bm25_results: list[tuple[str, float]],
        faiss_results: list[tuple[str, float]],
        top_k: int = 30
    ) -> list[tuple[str, float]]:
        """
        Input:

        bm25_results:
        [
            (chunk_id, score),
            ...
        ]

        faiss_results:
        [
            (chunk_id, score),
            ...
        ]

        Output:
        [
            (chunk_id, rrf_score),
            ...
        ]
        """

        scores = defaultdict(float)

        # BM25
        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            scores[chunk_id] += 1 / (self.k + rank)

        # FAISS
        for rank, (chunk_id, _) in enumerate(faiss_results, start=1):
            scores[chunk_id] += 1 / (self.k + rank)

        results = list(scores.items())

        results.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return results[:top_k]