"""建索引和检索。"""

import hashlib
import json
import os

import numpy as np

from bm25 import BM25
from chunker import split_text
from config import INDEX_DIR, TOP_K
from embedder import encode, encode_query

# BM25 的统计只跟文档块有关，加载一次之后复用
_bm25 = None


def _file_hash(text):
    """文件内容的指纹，内容变一个字符它就会变。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _load_old():
    """读旧索引和元数据，没有就返回空。"""
    try:
        vectors = np.load(os.path.join(INDEX_DIR, "vectors.npy"))
        with open(os.path.join(INDEX_DIR, "chunks.json"), encoding="utf-8") as f:
            data = json.load(f)
        with open(os.path.join(INDEX_DIR, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        return vectors, data["chunks"], data["sources"], meta
    except (FileNotFoundError, ValueError, KeyError):
        return None, [], [], {}


def build_index(docs):
    """docs 是 [(文件名, 文本)]，切块、向量化、保存到磁盘。

    增量逻辑：文件哈希没变就复用旧向量，只编码新增或修改过的文件。
    第一次运行没有元数据，会全量编码；之后就只有变化的文件会重算。
    """
    global _bm25

    old_vectors, old_chunks, old_sources, old_meta = _load_old()

    chunks = []
    sources = []
    vectors = []
    meta = {}

    for name, text in docs:
        digest = _file_hash(text)
        meta[name] = digest

        if old_meta.get(name) == digest:
            indexes = [i for i, source in enumerate(old_sources) if source == name]
            if indexes:
                for i in indexes:
                    chunks.append(old_chunks[i])
                    sources.append(name)
                    vectors.append(old_vectors[i])
                print(f"  {name}：内容没变，复用 {len(indexes)} 块")
                continue

        pieces = split_text(text)
        if not pieces:
            continue
        new_vectors = encode(pieces)
        chunks.extend(pieces)
        sources.extend([name] * len(pieces))
        vectors.extend(new_vectors)
        print(f"  {name}：重新编码 {len(pieces)} 块")

    if not chunks:
        print("没有可建索引的内容")
        return

    matrix = np.vstack(vectors)
    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(os.path.join(INDEX_DIR, "vectors.npy"), matrix)
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"chunks": chunks, "sources": sources},
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(os.path.join(INDEX_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 索引变了，关键词检索的统计也要跟着重建
    _bm25 = None
    print(f"索引已保存：{len(chunks)} 块，向量维度 {matrix.shape[1]}")


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


def search(question, top_k=TOP_K, rerank=False):
    """检索。

    rerank=False 时返回混合检索的结果；rerank=True 时先召回全部候选，
    再用重排序模型精排出前 top_k 条。
    """
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
    order = sorted(combined, key=combined.get, reverse=True)
    candidates = [(float(combined[i]), chunks[i], sources[i]) for i in order]

    if rerank:
        from reranker import rerank as do_rerank

        return do_rerank(question, candidates, top_k)

    return candidates[:top_k]
