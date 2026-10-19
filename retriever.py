"""建索引和检索。"""

import json
import os

import numpy as np

from bm25 import BM25
from chunker import split_text
from config import INDEX_DIR, TOP_K
from embedder import encode, encode_query

# BM25 的统计只跟文档块有关，加载一次之后复用
_bm25 = None


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


def rrf_merge(*rank_lists, k=60):
    """倒数排名融合（Reciprocal Rank Fusion）。

    两路检索的分数不在一个量纲上，直接相加不公平。RRF 不看分数只看排名：
    某个块在某一路排第 r 名，就给它 1/(k+r) 分，两路加起来。
    k 取 60 是常见经验值，作用是让排名靠前的差距别太夸张。
    """
    combined = {}
    for scores in rank_lists:
        order = np.argsort(scores)[::-1]
        for rank, index in enumerate(order, start=1):
            combined[int(index)] = combined.get(int(index), 0.0) + 1.0 / (k + rank)
    return combined


def search(question, top_k=TOP_K):
    """混合检索：向量路 + 关键词路，按排名融合后取前几块。"""
    global _bm25

    vectors, chunks, sources = load_index()
    if _bm25 is None:
        _bm25 = BM25(chunks)

    # 第一路：语义相似度
    query_vec = encode_query(question)
    vector_scores = vectors @ query_vec

    # 第二路：关键词打分
    keyword_scores = np.array(_bm25.scores(question))

    combined = rrf_merge(vector_scores, keyword_scores)
    order = sorted(combined, key=combined.get, reverse=True)[:top_k]

    return [(float(combined[i]), chunks[i], sources[i]) for i in order]
