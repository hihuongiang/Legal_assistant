"""Command-line entry point for effective-date legal corpus builds."""

from __future__ import annotations

import argparse
import sys

from src.parser.corpus_builder import build_effective_corpus
from src.parser.models import CorpusValidationError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an effective-date legal corpus from raw Parquet.")
    parser.add_argument("--source", default="data/Legal_Docs_Full_Raw_HTML.parquet")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--as-of", required=True, help="Effective date in YYYY-MM-DD format.")
    arguments = parser.parse_args(argv)
    try:
        manifest = build_effective_corpus(arguments.source, arguments.output_dir, arguments.as_of)
    except (CorpusValidationError, FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"Built {manifest.document_count} effective documents and {manifest.chunk_count} chunks "
        f"as of {manifest.as_of_date}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
