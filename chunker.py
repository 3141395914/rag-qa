"""把长文档切成小块。

为什么要切：整篇文档太长，塞不进模型上下文；就算塞得进，检索时也没法
精确命中某一段。切成几百字一块，检索才能找到最相关的那几块。

为什么要重叠：切点可能正好落在句子中间，让相邻两块重叠一点，
上下文不会断掉。
"""

from config import CHUNK_OVERLAP, CHUNK_SIZE


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """按字符切块，切点尽量落在句号或换行处。"""
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # 没到结尾时，往前找最近的句号或换行，避免把句子拦腰截断
        if end < len(text):
            window_start = start + chunk_size // 2
            cut = max(
                text.rfind("。", window_start, end),
                text.rfind("\n", window_start, end),
            )
            if cut != -1:
                end = cut + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break
        start = end - overlap

    return chunks
