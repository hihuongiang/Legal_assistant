"""Evaluate the legal QA engine against the offline development set."""

from __future__ import annotations

from pathlib import Path
import sys

from src.engine.legal_qa_engine import LegalQAEngine
from src.evaluation.evaluator import Evaluator


PROCESSED_DIR = Path("data/processed")
EVALUATION_PATH = Path("data/mock_dev_set.csv")
CHUNKS_PATH = PROCESSED_DIR / "effective_legal_chunks.json"
INDEX_PATH = PROCESSED_DIR / "effective_legal_chunks.faiss"
INDEX_MANIFEST_PATH = PROCESSED_DIR / "effective_legal_chunks.manifest.json"


def configure_utf8() -> None:
    """Use UTF-8 for Vietnamese command output where the console supports it."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def main() -> int:
    configure_utf8()
    try:
        engine = LegalQAEngine(
            chunks_path=str(CHUNKS_PATH),
            faiss_index_path=str(INDEX_PATH),
            faiss_metadata_path=str(INDEX_MANIFEST_PATH),
            top_k_bm25=30,
            top_k_dense=30,
            top_k_rrf=40,
            top_k_rerank=5,
        )
        summary = Evaluator(engine).run_evaluation(EVALUATION_PATH)
    except Exception as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    print(f"Evaluated {summary['count']} rows; mean recall: {summary['mean_recall']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
