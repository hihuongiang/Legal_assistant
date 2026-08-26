# Effective Legal Corpus Rebuild Design

## Goal

Rebuild the Legal QA corpus from data/Legal_Docs_Full_Raw_HTML.parquet so retrieval and citations use documents effective on a declared date and run safely on the existing 4 GB GPU.

## Effective-date policy

Parquet is the authoritative source. Build accepts --as-of YYYY-MM-DD, defaulting to the local date. Include a row when its ISO effFrom is on or before as_of and its normalized status is not explicitly expired, repealed, suspended, or cancelled.

Do not exclude a row only because the crawl-time status is Chưa có hiệu lực. In the current snapshot, six such rows have effFrom 2026-07-01 and are eligible on 2026-08-27; the two rows with 2027 dates are excluded. Record as_of and filter counts in the build manifest.

## Data and index contract

Deterministically deduplicate docs_code, preferring non-empty then longest HTML. Parse HTML to bounded article/clause chunks in data/processed/effective_legal_chunks.json. Each LegalChunk preserves chunk_id, law_id, law_name, article_name, clause_name, effective_date, status, source_url and content.

Split oversized clauses by paragraphs, sentences, then words to 350 words with a 40-word overlap. Build data/processed/effective_legal.faiss and effective_legal.manifest.json only from this corpus. Manifest binds IDs, corpus hash, source hash, as_of, vector count, dimension and model. Engine rejects any mismatch before search.

## CUDA runtime

Embedding and reranking are CUDA-only, FP16, sequence length 512 and batch size 1. Release the dense model before loading the reranker, and release the reranker before Ollama generation. No CPU fallback; CUDA/OOM errors exit non-zero with operation and configuration details. Ollama uses keep_alive=0.

## Output and acceptance

Citations come only from canonical metadata and retain the current results schema. Submission writes atomically. Tests use fakes and run offline. A build as of 2026-08-27 includes the six date-eligible stale-status rows and excludes the two 2027 rows; a CUDA smoke test either completes one question or fails clearly without partial output.

## Non-goals

No re-crawl, amendment inference not represented in Parquet, or historical-law retrieval is part of this change.
