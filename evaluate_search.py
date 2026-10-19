"""检索评测：看正确答案能不能被排到前面。

跑法：python evaluate_search.py

每个问题在 data/search_eval.json 里标注了正确段落的特征词，
脚本分别用纯向量和混合检索各跑一遍，比较正确段落的排名。

Recall@K：正确答案落在前 K 名里的比例。
MRR：正确段落排名的倒数取平均，排第 1 得 1 分，第 2 得 0.5 分。
"""

import json
import os

import numpy as np

from bm25 import BM25
from embedder import encode_query
from retriever import load_index, rrf_merge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_PATH = os.path.join(BASE_DIR, "data", "search_eval.json")


def rank_of_answer(order, chunks, key):
    """正确段落在排序里的名次，从 1 开始；找不到返回 None。"""
    for rank, index in enumerate(order, start=1):
        if key in chunks[index]:
            return rank
    return None


def evaluate(questions, vectors, chunks):
    bm25 = BM25(chunks)
    vector_ranks = []
    hybrid_ranks = []

    for item in questions:
        question = item["question"]
        key = item["answer_key"]
        query_vec = encode_query(question)

        vector_scores = vectors @ query_vec
        vector_order = np.argsort(vector_scores)[::-1]
        vector_ranks.append(rank_of_answer(vector_order, chunks, key))

        keyword_scores = np.array(bm25.scores(question))
        combined = rrf_merge(vector_scores, keyword_scores)
        hybrid_order = sorted(combined, key=combined.get, reverse=True)
        hybrid_ranks.append(rank_of_answer(hybrid_order, chunks, key))

    return vector_ranks, hybrid_ranks


def summarize(name, ranks):
    total = len(ranks)
    mrr = 0.0
    hits = {1: 0, 3: 0, 5: 0}

    for rank in ranks:
        if rank is None:
            continue
        for k in hits:
            if rank <= k:
                hits[k] += 1
        mrr += 1.0 / rank

    print(f"\n{name}")
    print(f"  Recall@1: {hits[1] / total:.1%}")
    print(f"  Recall@3: {hits[3] / total:.1%}")
    print(f"  Recall@5: {hits[5] / total:.1%}")
    print(f"  MRR:      {mrr / total:.3f}")


def main():
    with open(EVAL_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    vectors, chunks, sources = load_index()
    vector_ranks, hybrid_ranks = evaluate(questions, vectors, chunks)

    print(f"共 {len(questions)} 个问题，索引 {len(chunks)} 块")
    for item, v_rank, h_rank in zip(questions, vector_ranks, hybrid_ranks):
        print(f"  向量第 {v_rank} 名 | 混合第 {h_rank} 名  ← {item['question']}")

    summarize("纯向量检索", vector_ranks)
    summarize("混合检索", hybrid_ranks)


if __name__ == "__main__":
    main()
