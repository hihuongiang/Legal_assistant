# Effective Legal Corpus Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Produce and use a corpus built from the raw Parquet file, filtered by legal effective date, with validated FAISS artifacts and safe CUDA execution.

**Architecture:** A corpus builder produces canonical metadata-rich chunks and a build manifest. FAISS persists a matching index manifest. Engine startup validates both; each query uses one GPU transformer at a time and formats citations from the same canonical records.

**Tech Stack:** Python 3.11, pandas, pyarrow, BeautifulSoup, PyTorch CUDA, sentence-transformers, FAISS, rank-bm25, PyVi, Ollama, pytest.

**Spec:** docs/superpowers/specs/2026-08-27-legal-qa-stabilization-design.md

## Global constraints

- Source is data/Legal_Docs_Full_Raw_HTML.parquet.
- Include effFrom <= --as-of unless status explicitly states expired/repealed/suspended/cancelled.
- Build date defaults to the local date and is recorded in manifests.
- CUDA-only: FP16, 512 tokens, batch size 1, no silent CPU fallback.
- Never commit data/processed, raw data, results, logs or caches.

---

### Task 1: Add reproducible tooling and test fixtures

**Files:**
- Create: requirements.txt, .gitignore, pytest.ini
- Create: tests/conftest.py, tests/fixtures/raw_documents.json, tests/test_project_contract.py
- Modify: README.md

**Interfaces:** produces raw_document_frame fixture and an offline pytest suite.

- [ ] **Step 1: Write the failing fixture test**

    def test_fixture_has_stale_and_future_rows(raw_document_rows):
        assert {row["docs_code"] for row in raw_document_rows} == {"A/2026/QH", "B/2027/ND"}

    def test_readme_has_build_command():
        assert "main_build_corpus.py --as-of 2026-08-27" in Path("README.md").read_text(encoding="utf-8")

- [ ] **Step 2: Run it**

  Run: .venv\Scripts\python.exe -m pytest tests/test_project_contract.py -q

  Expected: FAIL because the harness does not exist.

- [ ] **Step 3: Implement the harness**

  Add pinned direct dependencies: torch CUDA 12.1, sentence-transformers, transformers 4.44 or newer but below 5, faiss-gpu, pandas, pyarrow, beautifulsoup4, rank-bm25, pyvi, ollama, tqdm and pytest. Add testpaths = tests. Ignore .venv, caches, data/processed, results.json, submission.zip and logs. Fixture rows must include stale 2026-07-01 and future 2027-01-01 effective dates.

- [ ] **Step 4: Document install/build/index/test commands and --as-of in README**

- [ ] **Step 5: Run the test again**

  Run: .venv\Scripts\python.exe -m pytest tests/test_project_contract.py -q

  Expected: PASS.

- [ ] **Step 6: Commit**

    git add requirements.txt .gitignore pytest.ini README.md tests
    git commit -m "test: establish legal corpus test harness"

### Task 2: Define and validate canonical chunks

**Files:**
- Create: src/parser/models.py
- Modify: src/parser/chunk_builder.py
- Create: tests/test_parser_models.py

**Interfaces:** LegalChunk.from_dict(record, record_index) -> LegalChunk; load_chunks(path) -> list[LegalChunk].

- [ ] **Step 1: Write failing metadata tests**

    def test_chunk_retains_citation_metadata():
        chunk = LegalChunk.from_dict({
            "chunk_id": "59-2020-QH14_D1_K1", "law_id": "59/2020/QH14",
            "law_name": "Luật Doanh nghiệp", "article_name": "Điều 1",
            "clause_name": "Khoản 1", "effective_date": "2021-01-01",
            "status": "Còn hiệu lực", "source_url": "https://example.test",
            "content": "Nội dung",
        }, 0)
        assert chunk.article_name == "Điều 1"

    def test_missing_field_identifies_record():
        with pytest.raises(CorpusValidationError, match=r"record 3.*law_name"):
            LegalChunk.from_dict({"chunk_id": "x"}, 3)

- [ ] **Step 2: Run it**

  Run: .venv\Scripts\python.exe -m pytest tests/test_parser_models.py -q

  Expected: FAIL because models.py is absent.

- [ ] **Step 3: Implement a frozen LegalChunk**

  Fields are chunk_id, law_id, law_name, article_name, clause_name, effective_date, status, source_url and content. Required fields reject empty strings. Add CorpusValidationError and JSON-serializable CorpusBuildManifest with as_of_date, source hash, counts and corpus hash.

- [ ] **Step 4: Update loader validation**

  Replace the existing local dataclass. Require a JSON list; validate every item; reject duplicates while reporting both indexes.

- [ ] **Step 5: Run and commit**

  Run: .venv\Scripts\python.exe -m pytest tests/test_parser_models.py -q

  Expected: PASS.

    git add src/parser/models.py src/parser/chunk_builder.py tests/test_parser_models.py
    git commit -m "feat: define canonical legal chunk contract"

### Task 3: Build an effective-date corpus from raw Parquet

**Files:**
- Create: src/parser/corpus_builder.py, main_build_corpus.py, tests/test_corpus_builder.py
- Retire: src/parser/c.py after replacement tests pass

**Interfaces:** select_effective_documents(frame, as_of) -> DataFrame; ArticleChunker.chunk_document(row) -> list[LegalChunk]; build_effective_corpus(source, output_dir, as_of) -> CorpusBuildManifest.

- [ ] **Step 1: Write failing filter tests**

    def test_stale_status_included_after_effective_date(raw_document_frame):
        actual = select_effective_documents(raw_document_frame, date(2026, 8, 27))
        assert actual.docs_code.tolist() == ["A/2026/QH"]

    def test_future_document_excluded(raw_document_frame):
        actual = select_effective_documents(raw_document_frame, date(2026, 8, 27))
        assert "B/2027/ND" not in actual.docs_code.tolist()

- [ ] **Step 2: Run it**

  Run: .venv\Scripts\python.exe -m pytest tests/test_corpus_builder.py -q

  Expected: FAIL because corpus_builder.py is absent.

- [ ] **Step 3: Implement selection and deduplication**

  Require docs_code, docs_title, source_url, issue_date, effFrom, status and html_content. Parse dates only as %Y-%m-%d; reject invalid dates. Exclude future rows and explicit statuses Hết hiệu lực, Bị bãi bỏ, Ngưng hiệu lực and Đình chỉ. Deduplicate docs_code by non-empty HTML, longest HTML, then source order.

- [ ] **Step 4: Implement parser and bounded chunking**

  Move the useful BeautifulSoup cleanup from c.py. Parse Điều, clauses and points; set metadata from row fields. Split clauses over 350 whitespace words by paragraphs, sentences, then words, adding 40-word overlap. IDs use normalized document code plus _D, _K and _P components. Reject empty chunks, chunks over 390 words and duplicate IDs.

- [ ] **Step 5: Write corpus and manifest atomically**

  Write data/processed/effective_legal_chunks.json and effective_legal_corpus.manifest.json through temporary sibling files. Manifest includes source SHA-256, as_of, input/deduplicated/included/excluded/chunk counts and corpus SHA-256. CLI supports --source, --output-dir and --as-of.

- [ ] **Step 6: Run and commit**

  Run: .venv\Scripts\python.exe -m pytest tests/test_corpus_builder.py tests/test_parser_models.py -q

  Expected: PASS for stale date, future date, deduplication, metadata and splitting.

    git add src/parser/corpus_builder.py main_build_corpus.py tests/test_corpus_builder.py src/parser/c.py
    git commit -m "feat: build effective legal corpus from parquet"

### Task 4: Bind FAISS to this corpus

**Files:**
- Create: src/retrieval/index_manifest.py, tests/test_faiss_manifest.py
- Modify: src/retrieval/faiss_store.py, main_build_index.py

**Interfaces:** IndexManifest.from_corpus(chunks, model_name, dimension, as_of_date); FaissStore.load(index_path, manifest_path, chunks) -> None; IndexValidationError.

- [ ] **Step 1: Write failing mismatch test**

    def test_index_rejects_different_corpus(tmp_path, canonical_chunks, alternate_chunks):
        store = FaissStore(FakeEmbedder(dimension=2))
        store.build_index(canonical_chunks)
        store.save(tmp_path / "index.faiss", tmp_path / "manifest.json", "2026-08-27")
        with pytest.raises(IndexValidationError, match="fingerprint"):
            FaissStore(FakeEmbedder(dimension=2)).load(
                tmp_path / "index.faiss", tmp_path / "manifest.json", alternate_chunks
            )

- [ ] **Step 2: Run it**

  Run: .venv\Scripts\python.exe -m pytest tests/test_faiss_manifest.py -q

  Expected: FAIL because no manifest contract exists.

- [ ] **Step 3: Implement JSON index manifest**

  Record schema version, ordered IDs, corpus hash, as_of, vector count, dimension and embedding model. On load verify index.ntotal, dimension, model, ID order, uniqueness and corpus fingerprint. Do not load legacy pickle metadata silently.

- [ ] **Step 4: Make index CLI use production paths only**

  Defaults: data/processed/effective_legal_chunks.json, effective_legal.faiss and effective_legal.manifest.json. Read corpus build manifest and propagate as_of. Reject non-processed paths without --allow-experimental-path.

- [ ] **Step 5: Run and commit**

  Run: .venv\Scripts\python.exe -m pytest tests/test_faiss_manifest.py -q

  Expected: PASS for valid load plus changed content, order, count and dimension.

    git add src/retrieval/index_manifest.py src/retrieval/faiss_store.py main_build_index.py tests/test_faiss_manifest.py
    git commit -m "feat: validate faiss index against corpus manifest"

### Task 5: Enforce safe CUDA model execution

**Files:**
- Create: src/runtime/gpu.py, tests/test_gpu_runtime.py
- Modify: src/embedding/dense_model.py, src/rerank/bge_reranker.py

**Interfaces:** require_cuda(); CudaRequiredError; GpuMemoryError; DenseEmbedder.close(); BGEReranker.close().

- [ ] **Step 1: Write failing GPU guard tests**

    def test_embedder_requires_cuda(monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        with pytest.raises(CudaRequiredError):
            DenseEmbedder()

    def test_oom_reports_limits():
        assert "batch_size=1" in str(GpuMemoryError("rerank", 1, 512))

- [ ] **Step 2: Run it**

  Run: .venv\Scripts\python.exe -m pytest tests/test_gpu_runtime.py -q

  Expected: FAIL because GPU error types are absent.

- [ ] **Step 3: Implement CUDA-only FP16 configuration**

  Require CUDA before model construction. Dense model uses CUDA, FP16, max sequence 512, batch 1, inference mode and autocast. Reranker uses CUDA, FP16, max length 512 and predict batch 1. Convert torch.OutOfMemoryError to GpuMemoryError carrying operation, batch and limit.

- [ ] **Step 4: Implement idempotent close**

  Delete model references, synchronize when possible and call torch.cuda.empty_cache.

- [ ] **Step 5: Run and commit**

  Run: .venv\Scripts\python.exe -m pytest tests/test_gpu_runtime.py -q

  Expected: PASS offline.

    git add src/runtime/gpu.py src/embedding/dense_model.py src/rerank/bge_reranker.py tests/test_gpu_runtime.py
    git commit -m "feat: enforce bounded cuda inference"

### Task 6: Validate startup and load one GPU model at a time

**Files:**
- Modify: src/engine/legal_qa_engine.py, src/llm/generator.py
- Create: tests/test_legal_qa_engine.py

**Interfaces:** LegalQAEngine validates corpus/index at startup; run_pipeline(query) retains answer and relevant_articles keys; LLMGenerator.ensure_available() -> None.

- [ ] **Step 1: Write failing lifecycle test**

    def test_dense_closes_before_reranker(monkeypatch, canonical_paths):
        events = []
        monkeypatch.setattr("src.engine.legal_qa_engine.DenseEmbedder", lambda: FakeDense(events))
        monkeypatch.setattr("src.engine.legal_qa_engine.BGEReranker", lambda: FakeReranker(events))
        engine = LegalQAEngine(**canonical_paths, llm=FakeLLM())
        engine.run_pipeline("Câu hỏi")
        assert events.index("dense.close") < events.index("reranker.create")

- [ ] **Step 2: Run it**

  Run: .venv\Scripts\python.exe -m pytest tests/test_legal_qa_engine.py -q

  Expected: FAIL because current code loads both models in constructor.

- [ ] **Step 3: Refactor model lifecycle**

  Constructor loads canonical chunks, BM25 and manifest-validated index only. search_top_40 constructs dense and closes it in finally. run_pipeline creates reranker only after dense closes, then closes reranker before prompt generation.

- [ ] **Step 4: Harden Ollama**

  ensure_available verifies qwen2.5:7b via ollama.list. generate passes keep_alive=0 and raises typed service errors rather than printing/continuing.

- [ ] **Step 5: Run and commit**

  Run: .venv\Scripts\python.exe -m pytest tests/test_legal_qa_engine.py -q

  Expected: PASS for lifecycle, stale index and unavailable Ollama.

    git add src/engine/legal_qa_engine.py src/llm/generator.py tests/test_legal_qa_engine.py
    git commit -m "feat: validate runtime corpus and gpu lifecycle"

### Task 7: Format citations and write results atomically

**Files:**
- Create: src/submission/__init__.py, src/submission/formatter.py
- Modify: main_submit.py, main_eval.py, src/evaluation/evaluator.py
- Create: tests/test_submission_formatter.py, tests/test_entry_points.py

**Interfaces:** format_submission_item(question, response, chunk_map) -> dict; write_submission_atomically(items, target) -> None.

- [ ] **Step 1: Write failing output tests**

    def test_submission_uses_canonical_metadata(canonical_chunk):
        item = format_submission_item(
            {"id": "q1", "question": "Q"},
            {"answer": "A", "relevant_articles": [canonical_chunk.chunk_id]},
            {canonical_chunk.chunk_id: canonical_chunk},
        )
        assert item["relevant_articles"][0].endswith("|Điều 1")

    def test_failed_write_keeps_old_file(tmp_path):
        target = tmp_path / "results.json"
        target.write_text("old", encoding="utf-8")
        with pytest.raises(RuntimeError):
            write_submission_atomically(FailingIterable(), target)
        assert target.read_text(encoding="utf-8") == "old"

- [ ] **Step 2: Run it**

  Run: .venv\Scripts\python.exe -m pytest tests/test_submission_formatter.py tests/test_entry_points.py -q

  Expected: FAIL because no formatter/atomic writer exists.

- [ ] **Step 3: Implement stable citation formatting**

  Use LegalChunk law_id, law_name and article_name; stable-sort/deduplicate document and article references; return empty references for empty retrieval.

- [ ] **Step 4: Implement safe entry points**

  Write to a temporary sibling then replace only after all inputs succeed. Remove temp files on error. Configure UTF-8, use production artifacts, print errors to stderr and exit 1. Evaluator returns count and mean recall.

- [ ] **Step 5: Run and commit**

  Run: .venv\Scripts\python.exe -m pytest tests/test_submission_formatter.py tests/test_entry_points.py -q

  Expected: PASS.

    git add src/submission main_submit.py main_eval.py src/evaluation/evaluator.py tests/test_submission_formatter.py tests/test_entry_points.py
    git commit -m "feat: generate validated atomic legal submissions"

### Task 8: Build and validate production artifacts

**Files:**
- Create: tests/test_real_corpus_contract.py
- Modify: README.md

**Interfaces:** generated data stays ignored; build output supplies auditable counts and hashes.

- [ ] **Step 1: Write real-artifact test**

    @pytest.mark.skipif(
        not Path("data/processed/effective_legal_chunks.json").exists(),
        reason="build corpus first",
    )
    def test_generated_chunks_are_unique_and_bounded():
        chunks = load_chunks("data/processed/effective_legal_chunks.json")
        assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
        assert all(len(chunk.content.split()) <= 390 for chunk in chunks)

- [ ] **Step 2: Run offline tests**

  Run: .venv\Scripts\python.exe -m pytest -q

  Expected: PASS; real-artifact test is skipped.

- [ ] **Step 3: Build corpus as of the requested date**

  Run:

    .venv\Scripts\python.exe main_build_corpus.py --source data/Legal_Docs_Full_Raw_HTML.parquet --output-dir data/processed --as-of 2026-08-27

  Expected: six 2026-07-01 stale-status documents are included; two 2027 documents are excluded; chunk IDs are unique.

- [ ] **Step 4: Build CUDA index**

  First ensure Ollama has no resident GPU model. Run:

    .venv\Scripts\python.exe main_build_index.py --corpus data/processed/effective_legal_chunks.json --output-dir data/processed

  Expected: batch-size-1 FP16 build completes; vector count equals manifest ID count.

- [ ] **Step 5: Run final validation and one-question smoke test**

  Run:

    .venv\Scripts\python.exe -m pytest -q
    .venv\Scripts\python.exe main_eval.py --limit 1

  Expected: tests pass. Smoke test either returns a citation-bearing response or exits non-zero with a specific CUDA/Ollama diagnostic and no partial result.

- [ ] **Step 6: Record evidence and commit source only**

  Add build date, corpus count, manifest fingerprint prefix and GPU settings to README; do not add generated files.

    git add README.md tests/test_real_corpus_contract.py
    git commit -m "docs: document effective corpus validation"

## Plan self-review

- Tasks 2–3 deliver raw-source filtering, metadata, deduplication and bounded chunks.
- Task 4 prevents corpus/index mismatch.
- Tasks 5–6 make the 4 GB CUDA path explicit and safe.
- Task 7 repairs citation/output failures.
- Tasks 1 and 8 provide reproducibility and verification.

