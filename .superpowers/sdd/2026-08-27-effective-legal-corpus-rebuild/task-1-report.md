# Task 1 report: reproducible tooling and test fixtures

## Changes

- Added `requirements.txt` with the requested CUDA 12.1 PyTorch source and
  direct runtime/test dependencies.
- Added `.gitignore` entries for the virtual environment, Python/pytest caches,
  generated processed data, result archives, and logs.
- Added `pytest.ini` to constrain discovery to `tests`.
- Added `raw_document_frame`, an offline session fixture that loads the two raw
  crawl rows from `tests/fixtures/raw_documents.json`.
- Added a project-contract test that verifies the required document IDs and the
  documented effective-date corpus command.
- Expanded the README with setup, corpus-build, index-build, test, and
  alternate `--as-of` instructions.

## TDD evidence

### RED

Command:

```powershell
pytest tests/test_project_contract.py -q
```

Output before the fixture and README command were added:

```text
FAILED tests/test_project_contract.py::test_raw_document_fixture_and_readme_contract
AssertionError: assert [] == ['A/2026/QH', 'B/2027/ND']
1 failed in 0.09s
```

The failure was expected: the raw document fixture did not exist, so the test
observed no document IDs.

### GREEN

Command:

```powershell
pytest tests/test_project_contract.py -q
```

Output:

```text
1 passed in 0.02s
```

## Full verification

Command:

```powershell
pytest -q
```

Output:

```text
1 passed in 0.02s
```

## Files

- `requirements.txt`
- `.gitignore`
- `pytest.ini`
- `tests/conftest.py`
- `tests/fixtures/raw_documents.json`
- `tests/test_project_contract.py`
- `README.md`

## Self-review

- The fixture is local JSON and the test suite makes no network/model calls.
- The two rows cover a stale crawl row effective on `2026-07-01` and a future
  row effective on `2027-01-01`.
- The fixture includes identifier, descriptive, issuer/date, crawl, source,
  HTML, and content fields for downstream corpus-building tests.
- No dependencies or models were installed/downloaded.
- Unrelated dirty worktree files were not staged.

## Concerns

- `main_build_corpus.py` is documented as the forthcoming corpus entry point;
  it is not introduced by this tooling-only task.
- The current offline suite contains one project-contract test; later tasks
  should add behavior-level corpus-build tests that consume this fixture.

## Fix round 1: review findings

### Root cause and changes

The original fixture used normalized, invented field names instead of the
Parquet schema consumed by the corpus pipeline, and `.gitignore` did not list
the observed generated legacy artifacts. The fixture now uses `docs_code`,
`docs_title`, `source_url`, `issue_date`, `effFrom`, `status`, and
`html_content`. The stale row has `status` `Chưa có hiệu lực` and `effFrom`
`2026-07-01`; the future row has `effFrom` `2027-01-01`.

The project-contract test now asserts every required field and those exact
edge values. It also runs `git check-ignore` to require `.ai-log` and each
legacy index/chunk artifact to be ignored while requiring
`data/Legal_Docs_Full_Raw_HTML.parquet` to remain visible to Git.

### RED

Command:

```powershell
pytest tests/test_project_contract.py -q
```

Output before the corrected fixture and ignore rules:

```text
FAILED tests/test_project_contract.py::test_raw_document_fixture_and_readme_contract
AssertionError: required Parquet fields were absent
FAILED tests/test_project_contract.py::test_gitignore_keeps_raw_parquet_and_ignores_legacy_generated_artifacts
AssertionError: [0, 1, 1, 1, 1, 1] != [0, 0, 0, 0, 0, 0]
2 failed in 0.58s
```

### GREEN and full verification

Focused command:

```powershell
pytest tests/test_project_contract.py -q
```

Output:

```text
2 passed in 0.44s
```

Full command:

```powershell
pytest -q
```

Output:

```text
2 passed in 0.35s
```
