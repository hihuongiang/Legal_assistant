# Legal Assistant

## Setup

Create a Python environment, then install the project dependencies. The
requirements file selects the CUDA 12.1 PyTorch wheel index for GPU builds.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Build the corpus

Build an effective-date corpus at a specific date. Documents whose effective
date is later than the supplied date are excluded.

```powershell
python main_build_corpus.py --as-of 2026-08-27
```

## Build the index

After the corpus is available, create or refresh the retrieval index.

```powershell
python main_build_index.py
```

## Run tests

The test suite is offline and uses local fixtures only.

```powershell
pytest
```

## Choose another effective date

Pass any ISO-8601 date through `--as-of` when rebuilding the corpus:

```powershell
python main_build_corpus.py --as-of YYYY-MM-DD
```
