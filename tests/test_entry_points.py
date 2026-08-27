"""Offline contracts for submission and evaluation command entry points."""

from pathlib import Path

from src.parser.models import LegalChunk


def legal_chunk() -> LegalChunk:
    return LegalChunk(
        chunk_id="chunk-1",
        law_id="59/2020/QH14",
        law_name="Law on Enterprises",
        article_name="Article 1",
        clause_name="",
        effective_date="2020-01-01",
        status="In force",
        source_url="https://example.test/law",
        content="Canonical content",
    )


def test_submit_entrypoint_uses_processed_artifacts_and_writes_formatted_output(tmp_path, monkeypatch):
    """Catches a submit command that reaches legacy corpus/index files or bypasses the formatter."""
    import json
    import main_submit

    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps([{"id": "q-1", "question": "What applies?"}]), encoding="utf-8"
    )
    output_path = tmp_path / "results.json"
    calls = []

    class FakeEngine:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.chunk_map = {"chunk-1": legal_chunk()}

        def run_pipeline(self, question):
            assert question == "What applies?"
            return {"answer": "The answer", "relevant_articles": ["chunk-1"]}

    monkeypatch.setattr(main_submit, "LegalQAEngine", FakeEngine)
    monkeypatch.setattr(main_submit, "QUESTIONS_PATH", questions_path)
    monkeypatch.setattr(main_submit, "RESULTS_PATH", output_path)

    assert main_submit.main() == 0
    assert Path(calls[0].pop("chunks_path")) == Path("data/processed/effective_legal_chunks.json")
    assert Path(calls[0].pop("faiss_index_path")) == Path("data/processed/effective_legal_chunks.faiss")
    assert Path(calls[0].pop("faiss_metadata_path")) == Path(
        "data/processed/effective_legal_chunks.manifest.json"
    )
    assert calls == [{"top_k_bm25": 30, "top_k_dense": 30, "top_k_rrf": 40, "top_k_rerank": 5}]
    assert json.loads(output_path.read_text(encoding="utf-8"))[0]["relevant_articles"] == [
        "59/2020/QH14|Law on Enterprises|Article 1"
    ]


def test_submit_entrypoint_reports_errors_to_stderr_and_returns_failure(tmp_path, monkeypatch, capsys):
    """Catches a missing input file being reported on stdout or treated as success."""
    import main_submit

    monkeypatch.setattr(main_submit, "QUESTIONS_PATH", tmp_path / "missing.json")

    assert main_submit.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing.json" in captured.err


def test_evaluator_returns_count_and_mean_recall_for_csv_rows(tmp_path):
    """Catches evaluation output that cannot be consumed programmatically after processing all rows."""
    from src.evaluation.evaluator import Evaluator

    csv_path = tmp_path / "evaluation.csv"
    csv_path.write_text("question,truth\nfirst,chunk-1\nsecond,chunk-2\n", encoding="utf-8")

    class FakeEngine:
        def run_pipeline(self, query):
            return {"answer": "offline", "relevant_articles": ["chunk-1"] if query == "first" else []}

    assert Evaluator(FakeEngine()).run_evaluation(csv_path) == {"count": 2, "mean_recall": 0.5}


def test_eval_entrypoint_reports_engine_errors_to_stderr_and_returns_failure(monkeypatch, capsys):
    """Catches an evaluation startup failure that leaves a successful process exit code."""
    import main_eval

    class FailingEngine:
        def __init__(self, **_kwargs):
            raise RuntimeError("processed corpus unavailable")

    monkeypatch.setattr(main_eval, "LegalQAEngine", FailingEngine)

    assert main_eval.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "processed corpus unavailable" in captured.err
