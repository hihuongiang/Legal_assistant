# Task 3 report: effective-date corpus builder

## Scope delivered

- Added `src/parser/corpus_builder.py` with the requested public APIs:
  `select_effective_documents`, `ArticleChunker.chunk_document`, and
  `build_effective_corpus`.
- Added `main_build_corpus.py` with `--source`, `--output-dir`, and `--as-of`.
  It prints document/chunk counts on success and exits non-zero through
  argparse for invalid input or build errors.
- Added `tests/test_corpus_builder.py` with offline Parquet fixtures covering
  effective-date and status selection, stable duplicate selection, required
  fields/date validation, article/clause metadata, the 30-adjacent-numeric-line
  table boundary, long-text overlap/limits, empty-result rejection, artifacts
  and hashes, and CLI behavior.
- Retired the obsolete untracked `src/parser/c.py` only after the replacement
  focused suite passed. Raw Parquet data and all unrelated dirty work were left
  unchanged. No real corpus artifacts were built.

## TDD evidence

### RED: new builder API

The replacement tests were added before `src/parser/corpus_builder.py` existed.

```text
$ python -m pytest tests/test_corpus_builder.py -q --basetemp=.pytest-tmp
FFFFFF                                                                   [100%]
ModuleNotFoundError: No module named 'src.parser.corpus_builder'
6 failed in 0.53s
```

This fails for the intended reason: no production builder module existed.

### RED: long-content boundary diagnosis

After the initial minimal implementation, the long-content test exposed an
actual boundary defect rather than a fixture issue:

```text
E       AssertionError: ... Left contains one more item: '45-2026-QH_D1_K1_P4'
1 failed, 5 passed in 2.08s
```

Root cause: the splitter pre-sliced the 730-word paragraph into 350-word units,
which forced the four-word article heading into its own chunk. The packer was
changed to defer word breaks until it can fill the current chunk.

### GREEN: builder

```text
$ python -m pytest tests/test_corpus_builder.py -q --basetemp=.pytest-tmp
......                                                                   [100%]
6 passed in 0.61s
```

### RED/GREEN: CLI

The entry point was removed before its test was added so the CLI behavior had a
genuine failing run.

```text
$ python -m pytest tests/test_corpus_builder.py::test_build_corpus_cli_reports_counts_and_rejects_invalid_dates -q --basetemp=.pytest-tmp
E       assert 2 == 0
E       can't open file '...\\main_build_corpus.py': [Errno 2] No such file or directory
1 failed in 0.64s
```

After recreating the minimal CLI wrapper:

```text
$ python -m pytest tests/test_corpus_builder.py -q --basetemp=.pytest-tmp
.......                                                                  [100%]
7 passed in 3.57s
```

### RED/GREEN: empty chunk prevention

The self-review identified an unstructured HTML edge case that returned no
chunks silently. The test was added first:

```text
$ python -m pytest tests/test_corpus_builder.py::test_article_chunker_rejects_html_that_produces_no_legal_chunks -q --basetemp=.pytest-tmp
E       Failed: DID NOT RAISE CorpusValidationError
1 failed in 0.66s
```

`ArticleChunker` now raises `CorpusValidationError` when no article boundary is
parsed. Final focused evidence:

```text
$ python -m pytest tests/test_corpus_builder.py -q --basetemp=.pytest-tmp
........                                                                 [100%]
8 passed in 3.71s
```

## Final verification

```text
$ python -m pytest -q --basetemp=.pytest-tmp
.............                                                            [100%]
13 passed in 3.84s

$ python -m compileall -q src\parser\corpus_builder.py main_build_corpus.py
# exit 0
```

Scoped whitespace checks (`git diff --check` for the retired file and
`git diff --no-index --check NUL` for each new file) reported no whitespace
errors. Repository-wide `git diff --check` still reports pre-existing trailing
whitespace in unrelated dirty files, which this task did not modify.

The test-only `.pytest-tmp` directory was removed after verification.

## Self-review

- Effective dates are parsed strictly with `%Y-%m-%d`; malformed values and
  missing required source columns raise `CorpusValidationError`.
- Dates on or before `as_of` remain eligible regardless of the pending status
  `Chưa có hiệu lực`; future dates and the four explicitly excluded statuses
  are filtered out. Duplicate document codes choose non-empty HTML, then the
  longest HTML, then earliest source order.
- The BeautifulSoup parser produces canonical `LegalChunk` metadata and IDs of
  the form `<normalized-doc-code>_D<article>_K<clause>[_P<part>]`. It tracks
  adjacent numeric lines across iterations, rejects zero parsed articles,
  rejects duplicate IDs, and bounds serialized content at 390 words.
- Content packing prefers paragraph/sentence units before breaking at word
  boundaries; follow-on pieces start with the prior 40 words.
- JSON and manifest files are written with temporary sibling files followed by
  `os.replace`. The manifest hash is calculated from the exact persisted JSON
  bytes and the source hash from the exact source Parquet bytes.

## Concerns

- The parser deliberately recognizes both English `Article` and Vietnamese
  `Điều` headings. Documents whose legal text uses another article-heading
  convention now fail explicitly instead of generating an empty corpus; adding
  another convention should be accompanied by a fixture and test.

## Review fix round

### Finding 1: exact ISO dates

`datetime.strptime(..., "%Y-%m-%d")` accepts non-zero-padded components on
this platform. A parameterized behavior test was added before the fix and
exercised `effFrom` and `as_of` independently for `2026-7-1`, `2026-07-1`,
and `2026-7-01`.

```text
$ python -m pytest tests/test_corpus_builder.py -q --basetemp=.pytest-tmp
..FFF....FF..                                                            [100%]
E       Failed: DID NOT RAISE CorpusValidationError
5 failed, 8 passed in 2.57s
```

`_parse_iso_date` now requires a full-match against `\d{4}-\d{2}-\d{2}` before
calling `strptime`, so calendar validation still comes from the standard date
parser while representation validation is exact.

### Finding 2: ordered chunk boundaries

Two public `ArticleChunker` tests were added first. The paragraph test uses two
200-word paragraphs that could fit only by splitting the second one; the
sentence test uses one 400-word paragraph made of two complete 200-word
sentences. Both assert the next boundary unit is wholly deferred to the next
chunk (after the 40-word overlap).

The same RED run above showed the current packer placed
`paragraph_two_000` and `sentence_two_000` in the preceding chunks. The packer
now preserves any complete paragraph or sentence that is at most 350 words,
starting a new core chunk when it does not fit. It breaks at word boundaries
only when that individual unit itself exceeds 350 words.

### GREEN and verification

```text
$ python -m pytest tests/test_corpus_builder.py -q --basetemp=.pytest-tmp
.............                                                            [100%]
13 passed in 1.97s

$ python -m pytest -q --basetemp=.pytest-tmp
..................                                                       [100%]
18 passed in 2.35s

$ python -m compileall -q src\parser\corpus_builder.py main_build_corpus.py
# exit 0
```

The test-only `.pytest-tmp` directory will be removed before committing. The
change remains limited to the builder, its behavior tests, and this report.
