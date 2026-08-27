# Task 8 production-artifact validation report

Date: 2026-08-27

## Source changes and commits

- `11af86d docs: document effective corpus validation`
  - Added `tests/test_real_corpus_contract.py`.
  - Added the initial observed validation record to `README.md`.
- `b029e26 fix: support year-only effective dates`
  - Added interval eligibility for `effFrom` values expressed as `YYYY`.
  - Added TDD coverage and corrected the README record after the retry.

Only source, tests, and README files were committed; this report is committed
as their audit record. No generated `data/processed` content was committed.
Commands 1–13 below are the historical pre-completion attempts, superseded by
the final completion record.

## Commands and observed output

1. `\.venv\Scripts\python.exe -m pytest tests\test_real_corpus_contract.py -q`

   Exit nonzero: `No module named pytest`.

2. `\.venv\Scripts\python.exe --version` and `-m pip show ...`

   The project environment is Python 3.11.9 and has pandas, pyarrow, Torch
   2.5.1+cu121, sentence-transformers, faiss-cpu, and Ollama. It does not have
   `pytest` or `faiss-gpu`. The host `py` interpreter is Python 3.13.1 and has
   pytest plus the dependencies necessary for the offline test suite.

3. `py -m pytest -q`

   Result: 34 passed, 1 skipped, 20 errors. Every error was a `PermissionError`
   creating `C:\Users\asus\AppData\Local\Temp\pytest-of-Huongiang`.

4. `py -m pytest -q --basetemp .pytest-tmp\task8-prebuild`

   Result: `54 passed, 1 skipped, 2 warnings`. The real-artifact test skipped
   because `data/processed/effective_legal_chunks.json` did not exist.

5. `\.venv\Scripts\python.exe main_build_corpus.py --source data\Legal_Docs_Full_Raw_HTML.parquet --output-dir data\processed --as-of 2026-08-27`

   Initial result: nonzero CLI diagnostic
   `effFrom at source row 13 must use YYYY-MM-DD, got '2025'`.

6. Raw Parquet inspection with pandas:

   - 3,044 rows total.
   - 38 values initially failed the strict ISO-only check.
   - Six expected stale-status records have `effFrom=2026-07-01`:
     `102/2025/QH15`, `106/2025/QH15`, `135/2025/QH15`, `109/2025/QH15`,
     `108/2025/QH15`, and `83/2025/TT-NHNN`.
   - Two future records have year 2027: `52/2024/NĐ-CP` and
     `62/2025/TT-NHNN`.

7. TDD red phase:

   `py -m pytest tests\test_corpus_builder.py -q --basetemp .pytest-tmp\task8-year-only-red-2`

   Result: 3 failed, 13 passed. All failures were caused by rejection of the
   `YYYY` effective-date value, including the now-specific ambiguous-interval
   assertion.

8. TDD green phase:

   `py -m pytest tests\test_corpus_builder.py -q --basetemp .pytest-tmp\task8-year-only-green`

   Result: `16 passed`. The tests prove that a year-only date is excluded
   before its interval, included after it, rejected when `as_of` falls inside
   it, and preserved unchanged in generated chunks.

9. Retried corpus build using the same command as step 5.

   Result: nonzero CLI diagnostic
   `effFrom at source row 255 must use YYYY-MM-DD, got ''`.

10. Follow-up raw Parquet inspection:

    `unsupported_effFrom=17`; all 17 remaining values are empty strings. The
    source rows are `58-TC/TCT`, `61 TC/TCDN`, `9-TC/TQÐ`,
    `02/1998/TT/BTC`, `06-TC/TCDN`, `23/1999/TT-BTC`, `8-TC/TCT`,
    `18-TC/TCT`, `62 /2004/TT-BTC`, `31/ 2004/TT-BTC`, `31-TC/HCSN`,
    `33-TC/CN`, `17-TC/NHKT`, `20-TC/CÐKT`, `40-TC/NN`, `02-TC-CĐKT`, and
    `03/2017/TT- BVHTTDL`.

11. `Test-Path data\processed` after both build attempts.

    Output: `data/processed: absent (no corpus artifacts written)`.

12. `py -m pytest -q --basetemp .pytest-tmp\task8-pre-commit`

    Result: `57 passed, 1 skipped, 2 warnings in 14.92s`. Warnings originate
    from FAISS SWIG extension types. `git diff --check` emitted no whitespace
    errors.

13. Pre-policy smoke attempt:

    `\.venv\Scripts\python.exe main_eval.py --limit 1`

    Result: exit 1, `Evaluation failed: [Errno 2] No such file or directory:
    'data\\processed\\effective_legal_chunks.json'`.

## Final artifact and runtime disposition

The authorized year-only policy was implemented without inventing an exact
day: `YYYY` is treated as `[YYYY-01-01, YYYY-12-31]`. A row is included only
when the interval end is at or before `as_of`, excluded before the interval,
and causes a `CorpusValidationError` when `as_of` lies inside the interval.
The raw value remains `LegalChunk.effective_date`.

The remaining 17 empty effective dates have no authorized or defensible
interval, so they are excluded and counted as `unknown legal effective date`.
The completed production build is recorded below. Generated corpus artifacts
exist locally and remain uncommitted; the separate FAISS index still requires a
stable embedding environment.

Legacy root-level artifact SHA-256 values were inspected, not overwritten:

- `data/faiss_index.bin`: `199E66E11501C76158EADBF69C482AC6758CDCD59199330C68E85EAB1784011C`
- `data/faiss_metadata.pkl`: `C1C3ED20EEF5F9B535DB7BD39663D7E2CA8AB969FA6039508072FAC10FC54E4D`
- `data/legal_chunks_v3.json`: `A6E1A084B056B628CBC1759EB357AA168F9F58AC96310389DFCF84B099846A95`
- `data/master_chunks.json`: `19D6C3484D05427849557F93AA76CAF3AAA0D9D9720E3E5CE502F67B26A65BF3`
- `data/master_chunks.jsonl`: `AF3246549E29B905A4FF59B4AA0BF34006143C36820A4CDC79D30322BCCEEE2A`

## Task 8 completion addendum (2026-08-27)

The duplicate-ID regression was completed after the initial report. A raw
document (`11/2026/TT-BCT`) repeats canonical article blocks, and 78 effective
records also use distinct raw `docs_code` spellings that normalize to the same
ID. The red test reproduced the old `CorpusValidationError: duplicate chunk
IDs in effective corpus`; the green tests now prove deterministic `_O2`
suffixes retain repeated clause content, including split and cross-document
occurrences. Occurrence tracking is document-scoped in `ArticleChunker` and
corpus-scoped in `build_effective_corpus`, with split `_P` suffixes preserved.

The completed production build command was:

    .venv\Scripts\python.exe main_build_corpus.py --source data\Legal_Docs_Full_Raw_HTML.parquet --output-dir data\processed --as-of 2026-08-27

It succeeded with 2,970 effective documents, 150,267 chunks, 17 unknown-date
exclusions, 589 fallback documents (6,572 chunks), and 29,756 duplicate clause
occurrences. Generated validation found 150,267 unique IDs, a maximum content
length of 390 words, and a corpus SHA-256 matching the manifest:
`a7907733779f22efa9275bebad8272125a197bca3112106140d1b3263109ac04`.

Fresh offline verification:

    py -m pytest -q --basetemp .pytest-tmp\task8-final

Result: `68 passed, 2 warnings in 149.30s`. The warnings are pre-existing FAISS
SWIG deprecation warnings.

Index generation was attempted after `ollama ps` confirmed no resident model
and `.venv` confirmed CUDA availability (`NVIDIA GeForce RTX 3050 Laptop GPU`).
It could not load `BAAI/bge-m3`: the environment denied network access to
`huggingface.co` (`WinError 10013`). The process exited with that diagnostic
before writing `effective_legal.faiss` or its index manifest. The post-index
smoke test was therefore not run. Legacy root-level artifacts were untouched;
generated `data/processed` outputs remain uncommitted.

## Post-review audit follow-up (2026-08-27)

Commit `38dcaa8` made selection accounting fully reconcilable and rebuilt the
same corpus hash. Its manifest records 3,044 raw rows, 3,025 eligible rows, 17
unknown-date exclusions, two future-date exclusions, zero inactive-status
exclusions, 55 duplicate-version exclusions, and 2,970 selected documents.
Fresh verification then passed `71` tests with the two pre-existing FAISS SWIG
deprecation warnings. The project virtual environment was repaired with
PyArrow 22.0.0, Transformers 4.46.3, Fsspec 2026.4.0, and pytest installed;
`pip check` passed without broken requirements.
