# Task 6 — Startup validation and sequential GPU lifecycle

## TDD evidence

- **RED:** `python -m pytest tests\\test_legal_qa_engine.py -q` initially
  produced four expected contract failures: no injectable LLM, no typed Ollama
  error, and no `keep_alive=0` request. The model lifecycle tests used fakes;
  no model, GPU, or network operation was invoked.
- **GREEN:** `python -m pytest -p no:asyncio tests\\test_legal_qa_engine.py -q`
  → **6 passed, 2 third-party SWIG deprecation warnings**.

## Changes

- Startup validates the canonical corpus, BM25 index, and manifest-bound FAISS
  index using BGE-M3's lightweight identity; it constructs neither CUDA model.
- Dense embedding is created only for FAISS search and is closed in `finally`.
  Reranking starts only afterwards and is closed before Ollama generation.
- Ollama preflight verifies `qwen2.5:7b`; failures are typed, and generation
  requests `keep_alive=0`.
- Removed engine import-time console output, which raised `UnicodeEncodeError`
  on the Windows CP1258 console. Runtime diagnostics remain in the pipeline.

## Full verification

`python -m pytest -p no:asyncio -q --basetemp .pytest-tmp\\task6-full-final` →
**47 passed, 2 third-party SWIG deprecation warnings** in 15.18s.

The explicit `--basetemp` is required in this environment because the default
Windows Temp directory denies pytest access.
