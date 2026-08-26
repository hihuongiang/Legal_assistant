from pathlib import Path
import subprocess

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_raw_document_fixture_and_readme_contract(raw_document_frame: pd.DataFrame):
    """Catches a missing or altered reproducible corpus-build contract."""
    expected_fields = {
        "docs_code",
        "docs_title",
        "source_url",
        "issue_date",
        "effFrom",
        "status",
        "html_content",
    }

    assert expected_fields <= set(raw_document_frame.columns)
    assert raw_document_frame["docs_code"].tolist() == ["A/2026/QH", "B/2027/ND"]
    policy_inputs = raw_document_frame.set_index("docs_code")
    assert policy_inputs.loc["A/2026/QH", "status"] == "Chưa có hiệu lực"
    assert policy_inputs.loc["A/2026/QH", "effFrom"] == "2026-07-01"
    assert policy_inputs.loc["B/2027/ND", "status"] == "Chưa có hiệu lực"
    assert policy_inputs.loc["B/2027/ND", "effFrom"] == "2027-01-01"
    assert "main_build_corpus.py --as-of 2026-08-27" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_gitignore_keeps_raw_parquet_and_ignores_legacy_generated_artifacts():
    """Catches rules that hide the raw source or retain generated artifacts."""
    ignored_paths = [
        ".ai-log/session.log",
        "data/faiss_index.bin",
        "data/faiss_metadata.pkl",
        "data/legal_chunks_v3.json",
        "data/master_chunks.json",
        "data/master_chunks.jsonl",
    ]
    ignored_return_codes = [
        subprocess.run(
            ["git", "check-ignore", "--quiet", path],
            cwd=ROOT,
            check=False,
        ).returncode
        for path in ignored_paths
    ]
    raw_input = subprocess.run(
        ["git", "check-ignore", "--quiet", "data/Legal_Docs_Full_Raw_HTML.parquet"],
        cwd=ROOT,
        check=False,
    )

    assert ignored_return_codes == [0] * len(ignored_paths)
    assert raw_input.returncode == 1
