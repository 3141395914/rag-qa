"""把长文档切成小块。

为什么要切：整篇文档太长，塞不进模型上下文；就算塞得进，检索时也没法
精确命中某一段。切成几百字一块，检索才能找到最相关的那几块。

为什么要重叠：切点可能正好落在句子中间，让相邻两块重叠一点，
上下文不会断掉。
"""

from config import CHUNK_OVERLAP, CHUNK_SIZE


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """切块。

    如果文档用 --- 分隔成一条条记录（比如岗位清单），每条单独成块，
    避免把不同岗位混在同一块里，检索出来的引用才准。
    """
    text = text.strip()
    if not text:
        return []

    if "\n---\n" in text:
        chunks = []
        for block in text.split("\n---\n"):
            block = block.strip()
            if not block:
                continue
            if len(block) <= chunk_size:
                chunks.append(block)
            else:
                chunks.extend(_split_long(block, chunk_size, overlap))
        return chunks

    return _split_long(text, chunk_size, overlap)


def _split_long(text, chunk_size, overlap):
    """长文本按字符切，切点尽量落在句号或换行处。"""
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
