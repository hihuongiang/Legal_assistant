# src/llm/prompts.py

from torch import chunk

from src.parser.chunk_builder import LegalChunk


SYSTEM_PROMPT = """
Bạn là trợ lý pháp lý Việt Nam.

NHIỆM VỤ:
- Chỉ trả lời dựa trên các điều luật được cung cấp trong CONTEXT.
- Không được tự bịa thêm thông tin.
- Không được viện dẫn điều luật không xuất hiện trong CONTEXT.
- Nếu CONTEXT không chứa đủ thông tin để trả lời, hãy trả lời:

"Không tìm thấy thông tin trong các điều luật được cung cấp."

YÊU CẦU:
- Trả lời ngắn gọn, chính xác.
- Ưu tiên trình bày dạng gạch đầu dòng nếu phù hợp.
- Luôn nêu rõ căn cứ pháp lý ở cuối câu trả lời.

Ví dụ:

Căn cứ:
- Luật Doanh nghiệp 2020, Điều 4 Khoản 4.
- Luật Hỗ trợ doanh nghiệp nhỏ và vừa 2017, Điều 16 Khoản 1.
"""


def format_context(chunks: list[LegalChunk]) -> str:
    """
    Chuyển list chunk thành context cho LLM.
    """

    contexts = []

    for i, chunk in enumerate(chunks, start=1):
        block = f"[CONTEXT {i}]\n{chunk.content}"
        contexts.append(block.strip())

    return "\n\n".join(contexts)


def build_prompt(
        query: str,
        chunks: list[LegalChunk]
) -> str:
    """
    Build prompt cho LLM.
    """

    context = format_context(chunks)

    prompt = f"""
{SYSTEM_PROMPT}

--------------------
CONTEXT

{context}

--------------------
CÂU HỎI

{query}

--------------------
YÊU CẦU

1. Chỉ sử dụng thông tin trong CONTEXT.
2. Không suy diễn ngoài CONTEXT.
3. Nếu không có thông tin thì trả lời:
"Không tìm thấy thông tin trong các điều luật được cung cấp."
4. Cuối câu trả lời phải ghi rõ căn cứ pháp lý.

TRẢ LỜI:
"""

    return prompt