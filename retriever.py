"""建索引和检索。"""

import json
import os

import numpy as np

from chunker import split_text
from config import INDEX_DIR, TOP_K
from embedder import encode, encode_query


def build_index(docs):
    """docs 是 [(文件名, 文本)]，切块、向量化、保存到磁盘。"""
    chunks = []
    sources = []

    for name, text in docs:
        for piece in split_text(text):
            chunks.append(piece)
            sources.append(name)

    if not chunks:
        print("文档是空的，没有可建索引的内容")
        return

    print(f"共切成 {len(chunks)} 块，开始向量化...")
    vectors = encode(chunks)

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(os.path.join(INDEX_DIR, "vectors.npy"), vectors)
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"chunks": chunks, "sources": sources},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"索引已保存：{len(chunks)} 块，向量维度 {vectors.shape[1]}")


def load_index():
    vectors = np.load(os.path.join(INDEX_DIR, "vectors.npy"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), encoding="utf-8") as f:
        data = json.load(f)
    return vectors, data["chunks"], data["sources"]


def search(question, top_k=TOP_K):
    """检索最相关的几块，返回 [(相似度, 文本, 来源), ...]。"""
    vectors, chunks, sources = load_index()
    query_vec = encode_query(question)

    # 两边都是单位向量，点积就等于余弦相似度
    scores = vectors @ query_vec
    order = np.argsort(scores)[::-1][:top_k]

    return [(float(scores[i]), chunks[i], sources[i]) for i in order]
