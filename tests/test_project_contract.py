from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_raw_document_fixture_and_readme_contract(raw_document_frame: pd.DataFrame):
    """Catches a missing or altered reproducible corpus-build contract."""
    assert raw_document_frame["document_id"].tolist() == ["A/2026/QH", "B/2027/ND"]
    assert "main_build_corpus.py --as-of 2026-08-27" in (ROOT / "README.md").read_text(encoding="utf-8")
