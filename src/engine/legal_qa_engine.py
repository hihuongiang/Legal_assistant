import os
from typing import Dict, Any, List

from src.parser.chunk_builder import LegalChunk, load_chunks

from src.embedding.dense_model import DenseEmbedder

from src.retrieval.bm25_retriever import BM25Retriever

from src.retrieval.faiss_store import FaissStore

from src.retrieval.rrf import RRF

from src.rerank.bge_reranker import BGEReranker

from src.llm.generator import LLMGenerator

from src.llm.prompts import build_prompt


class _StartupEmbeddingModel:
    """BGE-M3's published shape, used to validate an index without CUDA."""

    def get_sentence_embedding_dimension(self) -> int:
        return 1024


class _StartupEmbedder:
    """Minimal BGE-M3 identity accepted by ``FaissStore`` during startup."""

    model_name = "BAAI/bge-m3"

    def __init__(self):
        self.model = _StartupEmbeddingModel()


class LegalQAEngine:
    def __init__(
        self,
        chunks_path: str,
        faiss_index_path: str,
        faiss_metadata_path: str,
        top_k_bm25: int = 30,
        top_k_dense: int = 30,
        top_k_rrf: int = 40,
        top_k_rerank: int = 5,
        llm: LLMGenerator | None = None,
    ):
        """
        Khởi tạo toàn bộ pipeline AI.
        """
        self.top_k_bm25 = top_k_bm25
        self.top_k_dense = top_k_dense
        self.top_k_rrf = top_k_rrf
        self.top_k_rerank = top_k_rerank

        print("[1/7] Nạp kho dữ liệu chunks...")
        self.chunks = load_chunks(chunks_path)
        self.chunk_map: Dict[str, LegalChunk] = {
            chunk.chunk_id: chunk for chunk in self.chunks
        }

        print("[2/5] Xác thực FAISS Index...")
        self._index_embedder = _StartupEmbedder()
        self.faiss_store = FaissStore(self._index_embedder)
        if os.path.exists(faiss_index_path) and os.path.exists(faiss_metadata_path):
            self.faiss_store.load(faiss_index_path, faiss_metadata_path, self.chunks)
        else:
            print(" ---> [CẢNH BÁO]: Không tìm thấy FAISS index. Hãy chạy main_build_index.py trước.")

        print("[3/5] Khởi tạo BM25 Index...")
        self.bm25 = BM25Retriever()
        if self.chunks:
            self.bm25.build_index(self.chunks)

        print("[4/5] Khởi tạo RRF Fusion...")
        self.rrf = RRF(k=60)

        print("[5/5] Khởi tạo Qwen2.5-7B-Instruct (LLM)...")
        self.llm = llm or LLMGenerator()

        print(">>> HỆ THỐNG LEGAL QA ĐÃ SẴN SÀNG <<<")

    def search_top_40(self, query: str) -> List[LegalChunk]:
        """
        Thực thi Phase 1: Hybrid Retrieval + RRF
        """
        # 1. Tìm kiếm BM25
        bm25_results = self.bm25.search(query, top_k=self.top_k_bm25)

        # 2. Tìm kiếm FAISS.  The GPU model exists only for this operation.
        dense = DenseEmbedder()
        try:
            self.faiss_store.embedder = dense
            faiss_results = self.faiss_store.search(query, top_k=self.top_k_dense)
        finally:
            self.faiss_store.embedder = self._index_embedder
            dense.close()

        # 3. Trộn kết quả RRF
        rrf_results = self.rrf.fuse(
            bm25_results=bm25_results,
            faiss_results=faiss_results,
            top_k=self.top_k_rrf,
        )

        # 4. Map ID về Object LegalChunk
        candidate_chunks = []
        for chunk_id, _ in rrf_results:
            if chunk_id in self.chunk_map:
                candidate_chunks.append(self.chunk_map[chunk_id])

        return candidate_chunks

    def run_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Thực thi End-to-End: Trả về Dictionary chuẩn format
        """
        # ==========================================
        # BƯỚC 1: TRUY XUẤT (Retrieval)
        # ==========================================
        candidate_chunks = self.search_top_40(query)

        if not candidate_chunks:
            return {
                "answer": "Không tìm thấy thông tin trong các điều luật được cung cấp.",
                "relevant_articles": []
            }

        # ==========================================
        # BƯỚC 2: CHẤM ĐIỂM TINH (Reranking)
        # ==========================================
        reranker = BGEReranker(use_fp16=True)
        try:
            rerank_results = reranker.rerank(
                query=query,
                chunks=candidate_chunks,
                top_k=self.top_k_rerank,
            )
        finally:
            reranker.close()

        top_k_chunks = [chunk for chunk, score in rerank_results]
        top_k_ids = [chunk.chunk_id for chunk in top_k_chunks]

        # ==========================================
        # BƯỚC 3: SINH VĂN BẢN (Generation)
        # ==========================================
        # Gọi hàm build_prompt từ file prompts.py của bạn
        final_prompt = build_prompt(query=query, chunks=top_k_chunks)

        # Gọi generator chỉ với tham số prompt (đúng cấu trúc Ollama)
        llm_answer = self.llm.generate(final_prompt)

        # ==========================================
        # BƯỚC 4: GÁC CỔNG & ĐÓNG GÓI (Guardrail)
        # ==========================================
        # Nếu LLM trả lời không tìm thấy thông tin, làm rỗng mảng trích dẫn
        if "Không tìm thấy thông tin" in llm_answer:
            final_ids = []
        else:
            final_ids = top_k_ids

        return {
            "answer": llm_answer,
            "relevant_articles": final_ids
        }
