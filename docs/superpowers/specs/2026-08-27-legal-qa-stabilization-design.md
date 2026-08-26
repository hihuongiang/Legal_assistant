# Legal QA Stabilization Design

## Goal

Make the Legal QA pipeline reproducible and runnable on its existing 4 GB GPU by adopting one authoritative corpus, preserving legal metadata through retrieval, and validating every index/data pairing before inference or submission generation.

## Scope

This design covers the current Python retrieval and submission pipeline. It standardizes `data/master_chunks.json` as the production corpus and the existing FAISS index as its corresponding index. It does not migrate the application to `data/legal_chunks_v3.json`, rebuild the full corpus, or redesign answer quality/prompting beyond reliability safeguards.

## Decisions

- `data/master_chunks.json` is the sole production corpus.
- `data/faiss_index.bin` and `data/faiss_metadata.pkl` must be generated from the same ordered corpus as `master_chunks.json`.
- The application remains GPU-first. It does not silently fall back to CPU; insufficient VRAM produces an actionable error.
- The active `LegalChunk` model preserves all metadata required by `main_submit.py`: `law_id`, `law_name`, and `article_name`, in addition to the retrieval fields.
- `legal_chunks_v3.json` is treated as an experimental build artifact and cannot overwrite the production FAISS index.

## Data Contract

`LegalChunk` will be the single in-memory contract shared by loading, retrieval, reranking, evaluation, and submission:

```python
@dataclass(frozen=True)
class LegalChunk:
    chunk_id: str
    content: str
    law_id: str
    law_name: str
    article_name: str
    clause_name: str
    effective_date: str
    amends: list[str]
```

`load_chunks(path)` must reject invalid JSON records with a message naming the file, record index, and missing required fields. Optional metadata is normalized to safe empty values. Duplicate `chunk_id` values are rejected for the production corpus.

## Index Manifest and Validation

The FAISS metadata file will carry an `IndexManifest` rather than a bare list of IDs. It contains the ordered `chunk_ids`, corpus record count, vector count, embedding dimension, model name, and a SHA-256 fingerprint of ordered IDs plus content.

`FaissStore.load()` validates:

1. FAISS vector count equals manifest ID count.
2. Embedding dimension equals the active embedder dimension.
3. Manifest corpus count and fingerprint equal the currently loaded production corpus.
4. Chunk IDs are unique and retain their recorded order.

Any mismatch raises `IndexValidationError` before a search executes, with instructions to run the production index command. Legacy metadata that is a list is not accepted silently; it is rebuilt explicitly with the production command.

## GPU Execution Model

Both BGE-M3 embedding and BGE reranking run on CUDA. The application configures explicit device selection, FP16 where supported, fixed maximum sequence length, and conservative batch sizes.

- Embedder: CUDA, normalized vectors, `max_seq_length=512`, query batch size 1, indexing batch size initially 2.
- Reranker: CUDA, FP16, `max_length=512`, prediction batch size 1.
- Long text is truncated by the model limits rather than allowing attention allocation proportional to raw chunk length.
- CUDA out-of-memory exceptions are wrapped in `GpuMemoryError`, including the current operation, configured batch size, sequence limit, and remediation command/setting.

The production corpus is small enough to validate locally. Large experimental source data remains outside the production build path until separately cleaned and re-indexed.

## Pipeline Flow

```text
master_chunks.json
  -> load/validate LegalChunk records
  -> validate FAISS manifest against corpus
  -> BM25 + dense retrieval on GPU
  -> RRF
  -> GPU reranking
  -> prompt generation
  -> Ollama answer
  -> submission rows with preserved legal metadata
```

`main_build_index.py` takes an explicit corpus path, defaults to the production corpus, and writes the matching index/manifest pair. It refuses `legal_chunks_v3.json` unless an explicit experimental flag is supplied. `main_submit.py` uses only the metadata held on `LegalChunk`; it no longer assumes fields that the loader discarded.

## Errors and Observability

- Startup logs model/device, corpus count, index count, embedding dimension, and manifest fingerprint prefix.
- Missing Ollama service/model is raised before the first question with clear setup instructions.
- Invalid corpus, stale index, missing file, CUDA OOM, and failed LLM calls use distinct exception types and non-zero process exits.
- Vietnamese console output is UTF-8-safe through an application entry-point configuration, not per-shell manual environment variables.

## Tests and Acceptance Criteria

Automated tests will run with `pytest` and use fakes for embedding, reranking, and Ollama; they do not download models or require a GPU.

Required coverage:

1. Loader retains all legal metadata and rejects duplicate/malformed production records.
2. Index manifest detects vector-count, dimension, ID-order, and fingerprint mismatches.
3. Retrieval maps index IDs to the correct legal records.
4. Submission formatting succeeds and includes legal document/article references from `LegalChunk`.
5. GPU configuration uses CUDA defaults and turns `torch.OutOfMemoryError` into a clear domain error.
6. Entry points return non-zero on startup/pipeline failures.
7. Existing `results.json` schema remains compatible: `id`, `question`, `answer`, `relevant_docs`, `relevant_articles`.

Definition of done:

- `python -m pytest -q` passes without network/GPU access.
- Production index validation passes against `master_chunks.json` after a deliberate production index rebuild.
- A one-question GPU smoke test completes when BGE models and Ollama are locally available; otherwise it exits clearly before producing a partial submission.
- The repo includes pinned dependencies, documented setup/run/validation instructions, and ignores generated caches/logs/results.

## Non-goals and Follow-up

Cleaning `legal_chunks_v3.json`, repairing its duplicate ID, and re-chunking its oversized records are intentionally deferred. They require a separate data-quality project and a new index, rather than being mixed into this stabilization work.
