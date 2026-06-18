from dataclasses import dataclass, asdict
from typing import List
import json


@dataclass
class LegalChunk:
    chunk_id: str
    law_id: str
    law_name: str
    effective_date: str
    article_name: str
    clause_name: str
    content: str
    amends: str = ""


def load_chunks(filepath: str) -> List[LegalChunk]:
    """
    Đọc file JSON chứa danh sách chunk.

    Input:
        filepath: đường dẫn tới file json

    Output:
        List[LegalChunk]
    """

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []

    for item in data:
        chunk = LegalChunk(
            chunk_id=item["chunk_id"],
            law_id=item["law_id"],
            law_name=item["law_name"],
            effective_date=item.get("effective_date", ""),
            article_name=item["article_name"],
            clause_name=item["clause_name"],
            content=item["content"],
            amends=item.get("amends", "")
        )

        chunks.append(chunk)

    return chunks


def save_chunks(chunks: List[LegalChunk], filepath: str) -> None:
    """
    Lưu List[LegalChunk] thành file JSON.
    """

    data = [asdict(chunk) for chunk in chunks]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)