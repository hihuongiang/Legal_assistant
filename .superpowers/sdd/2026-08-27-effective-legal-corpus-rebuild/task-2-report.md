# Task 2 report: canonical chunks

## Scope

- Added `src/parser/models.py` with frozen `LegalChunk`, `CorpusValidationError`,
  and JSON-serializable `CorpusBuildManifest` (`to_dict`).
- Refactored `src/parser/chunk_builder.py` to load only JSON lists, validate each
  persisted record, and reject duplicate IDs while identifying both row indexes.
- Added `tests/test_parser_models.py` for retained citation metadata, required
  field diagnostics, and duplicate-ID rejection.

## TDD evidence

### RED

The test file was written before either production module existed.

```text
$ python -m pytest tests/test_parser_models.py -q
ERROR tests/test_parser_models.py
ModuleNotFoundError: No module named 'src.parser.models'
1 error in 1.01s
```

This is the expected pre-implementation failure: the canonical model module and
validation API did not exist.

### GREEN

After the minimal model and loader implementation:

```text
$ python -m pytest tests/test_parser_models.py -q --basetemp=.pytest-tmp
...                                                                      [100%]
3 passed in 0.35s
```

`--basetemp=.pytest-tmp` is required in this workspace because the default
Windows temporary directory denied pytest access. The initial run without it
executed two tests successfully and then failed only while creating `tmp_path`
(`PermissionError: [WinError 5]`).

## Final verification

```text
$ python -m pytest -q --basetemp=.pytest-tmp
.....                                                                    [100%]
5 passed in 1.65s
```

## Self-review

- `LegalChunk` is `frozen=True` and retains each required citation and source
  field. Required values are strings and cannot be blank; `clause_name` is
  trimmed and may be empty.
- Validation errors identify the offending record index and field. Duplicate
  IDs include both first and later row indexes.
- `load_chunks` rejects non-list JSON and malformed JSON with
  `CorpusValidationError` before downstream indexing can consume it.
- `CorpusBuildManifest.to_dict()` contains only strings and integers, so its
  result can be passed directly to `json.dumps`.
- Full-repository `git diff --check` reports pre-existing whitespace issues in
  unrelated dirty files; the Task 2 files themselves have no whitespace errors.
- No reviewer subagent was requested because this task explicitly prohibited
  dispatching subagents.
