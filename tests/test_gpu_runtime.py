"""Offline contract tests for bounded CUDA inference."""

import importlib
from contextlib import nullcontext

import numpy as np
import pytest
import torch


def _gpu_runtime():
    """Import lazily so a missing runtime module is reported as a test failure."""
    try:
        return importlib.import_module("src.runtime.gpu")
    except ModuleNotFoundError as error:
        pytest.fail(f"CUDA runtime module is missing: {error}")


def test_embedder_requires_cuda_before_constructing_a_model(monkeypatch):
    """Catches an accidental CPU fallback or model download on a CUDA-less host."""
    runtime = _gpu_runtime()
    from src.embedding.dense_model import DenseEmbedder

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "src.embedding.dense_model.SentenceTransformer",
        lambda *args, **kwargs: pytest.fail("model constructor must not run without CUDA"),
    )

    with pytest.raises(runtime.CudaRequiredError):
        DenseEmbedder()


def test_oom_reports_bounded_operation_context():
    """Catches opaque OOM failures that omit the safe inference limits."""
    runtime = _gpu_runtime()

    error = runtime.GpuMemoryError("rerank", 1, 512)

    assert error.operation == "rerank"
    assert error.batch_size == 1
    assert error.sequence_limit == 512
    assert "operation=rerank" in str(error)
    assert "batch_size=1" in str(error)
    assert "sequence_limit=512" in str(error)


class _DenseModel:
    def __init__(self):
        self.max_seq_length = None
        self.half_called = False
        self.encode_calls = []

    def half(self):
        self.half_called = True
        return self

    def encode(self, texts, **kwargs):
        self.encode_calls.append((texts, kwargs))
        return np.array([[1.0, 2.0]])


def test_embedder_uses_fp16_and_bounded_cuda_inference(monkeypatch):
    """Catches unsafe CUDA model setup or caller-controlled embedding batch sizes."""
    _gpu_runtime()
    from src.embedding.dense_model import DenseEmbedder

    model = _DenseModel()
    constructor_kwargs = {}
    autocast_kwargs = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("src.embedding.dense_model.torch.inference_mode", nullcontext)
    monkeypatch.setattr(
        "src.embedding.dense_model.torch.autocast",
        lambda **kwargs: (autocast_kwargs.update(kwargs) or nullcontext()),
    )

    def construct_model(*args, **kwargs):
        constructor_kwargs.update(kwargs)
        return model

    monkeypatch.setattr("src.embedding.dense_model.SentenceTransformer", construct_model)

    embedder = DenseEmbedder("offline-model")
    result = embedder.encode(["one"], batch_size=99)

    assert model.max_seq_length == 512
    assert model.half_called is True
    assert constructor_kwargs == {"device": "cuda"}
    assert autocast_kwargs == {"device_type": "cuda", "dtype": torch.float16}
    assert model.encode_calls == [
        (
            ["one"],
            {
                "batch_size": 1,
                "normalize_embeddings": True,
                "convert_to_numpy": True,
                "show_progress_bar": False,
            },
        )
    ]
    assert result.tolist() == [[1.0, 2.0]]


def test_embedder_translates_torch_oom_to_typed_context(monkeypatch):
    """Catches raw torch OOM exceptions leaking from embedding inference."""
    runtime = _gpu_runtime()
    from src.embedding.dense_model import DenseEmbedder

    model = _DenseModel()
    model.encode = lambda *args, **kwargs: (_ for _ in ()).throw(torch.OutOfMemoryError("oom"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "src.embedding.dense_model.SentenceTransformer", lambda *args, **kwargs: model
    )

    with pytest.raises(runtime.GpuMemoryError, match="operation=embed"):
        DenseEmbedder("offline-model").encode(["one"])


class _RerankerModel:
    def __init__(self):
        self.model = _DenseModel()
        self.predict_calls = []

    def predict(self, pairs, **kwargs):
        self.predict_calls.append((pairs, kwargs))
        return np.array([0.2, 0.8])


class _Chunk:
    def __init__(self, content):
        self.content = content


def test_reranker_uses_cuda_fp16_and_bounded_prediction(monkeypatch):
    """Catches reranking that can exceed the configured CUDA sequence or batch limit."""
    _gpu_runtime()
    from src.rerank.bge_reranker import BGEReranker

    model = _RerankerModel()
    constructor_kwargs = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def construct_model(*args, **kwargs):
        constructor_kwargs.update(kwargs)
        return model

    monkeypatch.setattr("src.rerank.bge_reranker.CrossEncoder", construct_model)

    chunks = [_Chunk("first"), _Chunk("second")]
    ranked = BGEReranker("offline-model").rerank("question", chunks)

    assert model.model.half_called is True
    assert constructor_kwargs == {"max_length": 512, "device": "cuda"}
    assert model.predict_calls == [
        (
            [("question", "first"), ("question", "second")],
            {"batch_size": 1},
        )
    ]
    assert [chunk.content for chunk, _score in ranked] == ["second", "first"]


def test_reranker_translates_torch_oom_to_typed_context(monkeypatch):
    """Catches raw torch OOM exceptions leaking from reranker prediction."""
    runtime = _gpu_runtime()
    from src.rerank.bge_reranker import BGEReranker

    model = _RerankerModel()
    model.predict = lambda *args, **kwargs: (_ for _ in ()).throw(torch.OutOfMemoryError("oom"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("src.rerank.bge_reranker.CrossEncoder", lambda *args, **kwargs: model)

    with pytest.raises(runtime.GpuMemoryError, match="operation=rerank"):
        BGEReranker("offline-model").rerank("question", [_Chunk("first")])


@pytest.mark.parametrize(
    ("module_path", "class_name", "constructor_name"),
    [
        ("src.embedding.dense_model", "DenseEmbedder", "SentenceTransformer"),
        ("src.rerank.bge_reranker", "BGEReranker", "CrossEncoder"),
    ],
)
def test_model_close_is_idempotent_and_releases_cuda_cache(
    monkeypatch, module_path, class_name, constructor_name
):
    """Catches cleanup that fails on a second call or leaves GPU cache unreleased."""
    _gpu_runtime()
    module = importlib.import_module(module_path)
    model = _RerankerModel() if class_name == "BGEReranker" else _DenseModel()
    synchronize_calls = []
    empty_cache_calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronize_calls.append(True))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))
    monkeypatch.setattr(module, constructor_name, lambda *args, **kwargs: model)

    instance = getattr(module, class_name)("offline-model")
    instance.close()
    instance.close()

    assert instance.model is None
    assert synchronize_calls == [True, True]
    assert empty_cache_calls == [True, True]
