"""Pure citation formatting and atomic persistence for legal submissions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from src.parser.models import LegalChunk


def format_submission_item(
    question: Mapping[str, Any],
    response: Mapping[str, Any],
    chunks_by_id: Mapping[str, LegalChunk],
) -> dict[str, Any]:
    """Return one schema-compatible result with stable canonical citations."""
    relevant_docs: list[str] = []
    relevant_articles: list[str] = []
    seen_docs: set[str] = set()
    seen_articles: set[str] = set()

    for chunk_id in response.get("relevant_articles", []):
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue

        document_citation = f"{chunk.law_id}|{chunk.law_name}"
        article_citation = f"{document_citation}|{chunk.article_name}"
        if document_citation not in seen_docs:
            seen_docs.add(document_citation)
            relevant_docs.append(document_citation)
        if article_citation not in seen_articles:
            seen_articles.add(article_citation)
            relevant_articles.append(article_citation)

    return {
        "id": question["id"],
        "question": question["question"],
        "answer": response["answer"],
        "relevant_docs": relevant_docs,
        "relevant_articles": relevant_articles,
    }


def write_submission(output_path: str | Path, items: Iterable[Mapping[str, Any]]) -> None:
    """Persist all results atomically, preserving an existing target on failure."""
    results = list(items)
    target = Path(output_path)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
