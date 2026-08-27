# Task 7: Citation formatting and safe submission

## Scope

- Added a pure submission formatter that derives stable, deduplicated document
  and article citations exclusively from `LegalChunk` metadata.
- Added an atomic UTF-8 JSON writer that materializes all items before creating
  a sibling temporary file and replacing the final target.
- Updated submission and evaluation entry points to configure UTF-8, use the
  canonical `data/processed` corpus/index/manifest paths, and return a nonzero
  exit code while reporting failures to stderr.
- Made `Evaluator.run_evaluation` return `count` and `mean_recall`.

## TDD evidence

### RED

```powershell
python -m pytest tests/test_submission_formatter.py tests/test_entry_points.py -q --basetemp .pytest-tmp/task7-red
```

Result before implementation: 7 failed. The failures identified the absent
`src.submission` package, missing canonical entry-point constants, and missing
evaluator/entry-point return contracts.

An additional error-path RED run changed the engine fake to raise
`RuntimeError`; it failed until both entry points handled all runtime errors
through stderr and exit status `1`.

### GREEN

```powershell
python -m pytest tests/test_submission_formatter.py tests/test_entry_points.py -q --basetemp .pytest-tmp/task7-final-focused
```

Result: `7 passed` (two pre-existing native FAISS SWIG deprecation warnings).

### Full verification

```powershell
python -m pytest -q --basetemp .pytest-tmp/task7-final-full-2
```

Result: `54 passed` (the same two native FAISS SWIG deprecation warnings).
