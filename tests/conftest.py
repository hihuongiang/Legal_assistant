import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def raw_document_frame() -> pd.DataFrame:
    """Representative raw crawl rows for offline corpus-build tests."""
    fixture_path = Path(__file__).parent / "fixtures" / "raw_documents.json"
    return pd.DataFrame(json.loads(fixture_path.read_text(encoding="utf-8")))
