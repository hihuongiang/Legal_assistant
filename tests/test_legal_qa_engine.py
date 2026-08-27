"""Offline lifecycle and Ollama contract tests for the legal QA engine."""

import importlib
import sys
from types import SimpleNamespace

import pytest


def test_engine_import_has_no_console_side_effects(capsys):
    """Catches import-time output that is unsafe on a Windows CP1258 console."""
    sys.modules.pop("src.engine.legal_qa_engine", None)
    importlib.import_module("src.engine.legal_qa_engine")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


class _FakeDense:
    def __init__(self, events):
        self.events = events
        self.events.append("dense.create")

    def close(self):
        self.events.append("dense.close")


class _FakeReranker:
    def __init__(self, events):
        self.events = events
        self.events.append("reranker.create")

    def rerank(self, query, chunks, top_k):
        self.events.append("reranker.rerank")
        return [(chunks[0], 1.0)]

    def close(self):
        self.events.append("reranker.close")


class _FakeFaissStore:
    def __init__(self, embedder, events):
        self.embedder = embedder
        self.events = events
        self.events.append("faiss.create")

    def load(self, index_path, manifest_path, chunks):
        self.events.append("faiss.load")
        self.loaded_chunks = chunks

    def search(self, query, top_k):
        self.events.append("faiss.search")
        return [("chunk-1", 0.9)]


class _FakeBM25:
    def __init__(self, events):
        self.events = events

    def build_index(self, chunks):
        self.events.append("bm25.build")

    def search(self, query, top_k):
        self.events.append("bm25.search")
        return [("chunk-1", 0.8)]


class _FakeRRF:
    def __init__(self, events, k):
        self.events = events

    def fuse(self, bm25_results, faiss_results, top_k):
        self.events.append("rrf.fuse")
        return [("chunk-1", 0.9)]


class _FakeLLM:
    def __init__(self, events):
        self.events = events

    def generate(self, prompt):
        self.events.append("llm.generate")
        return "Câu trả lời"


def test_engine_releases_dense_before_creating_reranker(monkeypatch):
    """Catches overlapping CUDA model lifetimes during a successful pipeline run."""
    from src.engine import legal_qa_engine as engine_module

    events = []
    chunk = SimpleNamespace(chunk_id="chunk-1", content="Nội dung luật")
    monkeypatch.setattr(engine_module, "load_chunks", lambda _path: [chunk])
    monkeypatch.setattr(engine_module, "DenseEmbedder", lambda: _FakeDense(events))
    monkeypatch.setattr(engine_module, "BGEReranker", lambda use_fp16: _FakeReranker(events))
    monkeypatch.setattr(
        engine_module,
        "FaissStore",
        lambda embedder: _FakeFaissStore(embedder, events),
    )
    monkeypatch.setattr(engine_module, "BM25Retriever", lambda: _FakeBM25(events))
    monkeypatch.setattr(engine_module, "RRF", lambda k: _FakeRRF(events, k))
    monkeypatch.setattr(engine_module.os.path, "exists", lambda _path: True)

    engine = engine_module.LegalQAEngine(
        chunks_path="chunks.json",
        faiss_index_path="index.faiss",
        faiss_metadata_path="index.manifest.json",
        llm=_FakeLLM(events),
    )
    result = engine.run_pipeline("Câu hỏi")

    assert events[:3] == ["faiss.create", "faiss.load", "bm25.build"]
    assert events.index("dense.close") < events.index("reranker.create")
    assert events.index("reranker.close") < events.index("llm.generate")
    assert result == {"answer": "Câu trả lời", "relevant_articles": ["chunk-1"]}


def test_engine_closes_dense_when_faiss_search_fails(monkeypatch):
    """Catches a failed dense search that leaves the CUDA embedder allocated."""
    from src.engine import legal_qa_engine as engine_module

    events = []
    chunk = SimpleNamespace(chunk_id="chunk-1", content="Nội dung luật")

    class FailingFaissStore(_FakeFaissStore):
        def search(self, query, top_k):
            self.events.append("faiss.search")
            raise RuntimeError("index failed")

    monkeypatch.setattr(engine_module, "load_chunks", lambda _path: [chunk])
    monkeypatch.setattr(engine_module, "DenseEmbedder", lambda: _FakeDense(events))
    monkeypatch.setattr(
        engine_module,
        "FaissStore",
        lambda embedder: FailingFaissStore(embedder, events),
    )
    monkeypatch.setattr(engine_module, "BM25Retriever", lambda: _FakeBM25(events))
    monkeypatch.setattr(engine_module, "RRF", lambda k: _FakeRRF(events, k))
    monkeypatch.setattr(engine_module.os.path, "exists", lambda _path: True)

    engine = engine_module.LegalQAEngine(
        "chunks.json",
        "index.faiss",
        "index.manifest.json",
        llm=_FakeLLM(events),
    )

    with pytest.raises(RuntimeError, match="index failed"):
        engine.search_top_40("Câu hỏi")

    assert events[-1] == "dense.close"


def test_llm_preflight_rejects_a_missing_configured_model(monkeypatch):
    """Catches Ollama connectivity checks that do not confirm the requested model."""
    from src.llm import generator

    monkeypatch.setattr(generator.ollama, "list", lambda: {"models": [{"name": "other:latest"}]})
    llm = generator.LLMGenerator()

    with pytest.raises(generator.OllamaModelUnavailableError, match="qwen2.5:7b"):
        llm.ensure_available()


def test_llm_preflight_wraps_an_invalid_service_response(monkeypatch):
    """Catches malformed list responses escaping as untyped client errors."""
    from src.llm import generator

    monkeypatch.setattr(generator.ollama, "list", lambda: object())

    with pytest.raises(generator.OllamaServiceError, match="preflight"):
        generator.LLMGenerator().ensure_available()


def test_llm_generation_releases_model_and_wraps_service_failures(monkeypatch):
    """Catches a persistent Ollama model or raw client exception escaping generation."""
    from src.llm import generator

    chat_calls = []
    monkeypatch.setattr(generator.ollama, "list", lambda: {"models": [{"name": "qwen2.5:7b"}]})
    monkeypatch.setattr(
        generator.ollama,
        "chat",
        lambda **kwargs: (chat_calls.append(kwargs) or {"message": {"content": "Trả lời"}}),
    )

    assert generator.LLMGenerator().generate("Câu hỏi") == "Trả lời"
    assert chat_calls[0]["keep_alive"] == 0

    monkeypatch.setattr(
        generator.ollama,
        "chat",
        lambda **_kwargs: (_ for _ in ()).throw(ConnectionError("offline")),
    )
    with pytest.raises(generator.OllamaServiceError, match="generation"):
        generator.LLMGenerator().generate("Câu hỏi")
