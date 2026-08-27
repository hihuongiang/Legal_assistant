"""Run the legal QA engine and write a submission in the required schema."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from src.engine.legal_qa_engine import LegalQAEngine
from src.submission.formatter import format_submission_item, write_submission


PROCESSED_DIR = Path("data/processed")
QUESTIONS_PATH = Path("data/R2AIStage1DATA.json")
RESULTS_PATH = Path("results.json")
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
        with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
            questions = json.load(file)
        if not isinstance(questions, list):
            raise ValueError("submission questions must be a JSON list")

        engine = LegalQAEngine(
            chunks_path=str(CHUNKS_PATH),
            faiss_index_path=str(INDEX_PATH),
            faiss_metadata_path=str(INDEX_MANIFEST_PATH),
            top_k_bm25=30,
            top_k_dense=30,
            top_k_rrf=40,
            top_k_rerank=5,
        )
        results = [
            format_submission_item(question, engine.run_pipeline(question["question"]), engine.chunk_map)
            for question in questions
        ]
        write_submission(RESULTS_PATH, results)
    except Exception as error:
        print(f"Submission failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
