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

## Production validation record (2026-08-27)

The raw Parquet source was inspected before the production build: it contains
3,044 rows, including the six `2026-07-01` rows that should be eligible and
two 2027 rows that should be excluded for the requested date. The builder
accepts 21 year-only `effFrom` values as uncertain calendar-year intervals;
it includes them only after the year and preserves their raw value in chunks.
The build did not produce a corpus or index because 17 source rows have an
empty `effFrom` value (the first is source row 255, `58-TC/TCT`). An empty
date has no safe effective-date interval, so the corpus contract rejects it
before writing `data/processed`; there is no generated chunk count,
fingerprint, or FAISS manifest to report.

CUDA indexing remains configured for one FP16 embedding at a time, with batch
size 1 and a 512-token limit. It must be run only after the raw-date quality
issue is resolved and the corpus build succeeds. The offline test suite passed
with 57 tests and one expected generated-artifact skip using a workspace-local
pytest temporary directory; the project virtual environment needs `pytest`
installed before it can run the documented test command.
