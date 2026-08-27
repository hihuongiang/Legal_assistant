# Task 5 — Safe CUDA model execution

## TDD evidence

- **RED:** `python -m pytest tests/test_gpu_runtime.py -q` → 8 failed because
  `src.runtime.gpu` did not exist. The tests use mocked model constructors, so
  this run did not construct, download, or execute a model.
- **GREEN:** `python -m pytest tests/test_gpu_runtime.py -q` → 8 passed. The
  only warnings are third-party SWIG deprecation warnings emitted while
  importing dependencies.

## Full verification

- `TEMP` and `TMP` were redirected to `.pytest-tmp` because the default Windows
  temporary directory denied pytest access.
- `$env:TEMP = (Resolve-Path '.pytest-tmp').Path; $env:TMP = $env:TEMP; python
  -m pytest -q` → **39 passed, 2 warnings** in 18.84s.

## Scope

CUDA availability is enforced before construction; both wrappers use CUDA FP16,
a 512-token limit, and batch size 1. Torch OOM exceptions become
`GpuMemoryError` with operation and limits. Both `close()` methods are
idempotent and release the model reference, synchronize where available, and
empty CUDA cache.

## P1 constructor OOM follow-up

- **RED:** mocked `SentenceTransformer` and `CrossEncoder` constructors raised
  `torch.OutOfMemoryError`; both raw errors escaped (2 failures).
- **GREEN:** constructors now release partial CUDA state and raise
  `GpuMemoryError` with the correct operation, batch size, and sequence limit.
  `python -m pytest tests/test_gpu_runtime.py -q` → **10 passed, 2 warnings**
  in 31.83s (third-party SWIG deprecation warnings only).
- **Full:** with `TEMP` and `TMP` redirected to `.pytest-tmp`, `python -m
  pytest -q` → **41 passed, 2 warnings** in 34.67s.
