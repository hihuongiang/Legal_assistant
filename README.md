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
python main_build_index.py --chunks data/processed/effective_legal_chunks.json --index data/processed/effective_legal_chunks.faiss --manifest data/processed/effective_legal_chunks.manifest.json --corpus-manifest data/processed/effective_legal_corpus.manifest.json
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
The build excludes 17 source rows with an empty `effFrom` value (the first is
source row 255, `58-TC/TCT`) as an auditable `unknown legal effective date`
case, two future-effective rows, and zero inactive-status rows. Of 3,044 raw
rows, 3,025 were eligible before 55 duplicate document versions were reduced
to 2,970 effective documents and 150,267 unique chunks. The corpus SHA-256
begins `a7907733779f22ef`; 589 documents
used the bounded `Toàn văn` fallback (6,572 chunks), and 29,756 repeated
canonical clause occurrences received deterministic occurrence suffixes.

CUDA indexing remains configured for one FP16 embedding at a time, with batch
size 1 and a 512-token limit. The offline suite passed with 70 tests and two
pre-existing FAISS SWIG deprecation warnings using a workspace-local pytest
temporary directory. The index must be built separately after the BAAI/bge-m3
model is available in the project environment.
