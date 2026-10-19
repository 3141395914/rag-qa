"""关键词检索（BM25）。

向量检索擅长"意思相近"，但查型号、人名、专有名词这类细节时容易模糊。
BM25 按字面词频打分，正好补上这个短板。

中文没有现成的分词器，这里用相邻两个字组成的 bigram 当词。
比如"招聘信息"会切成"招聘""聘信""信息"。这样不用装分词库也能用。
"""

import math
import re
from collections import Counter

CJK = re.compile(r"[\u4e00-\u9fff]+")
WORD = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text):
    """中文切两字一组，英文和数字按整词切。"""
    tokens = []
    for seq in CJK.findall(text):
        if len(seq) == 1:
            tokens.append(seq)
        else:
            tokens.extend(seq[i : i + 2] for i in range(len(seq) - 1))
    tokens.extend(word.lower() for word in WORD.findall(text))
    return tokens


class BM25:
    """经典 BM25 打分。k1 控制词频饱和，b 控制文档长度的影响。"""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(doc) for doc in docs]
        self.n = len(self.docs)
        self.avg_len = sum(len(d) for d in self.docs) / max(self.n, 1)
        self.tf = [Counter(d) for d in self.docs]

        df = Counter()
        for doc in self.docs:
            df.update(set(doc))
        self.idf = {
            term: math.log((self.n - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def scores(self, query):
        """返回每个文档的 BM25 分数。"""
        q_tokens = tokenize(query)
        result = []

        for i, tf in enumerate(self.tf):
            doc_len = len(self.docs[i])
            score = 0.0
            for term in q_tokens:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
                score += idf * freq * (self.k1 + 1) / denom
            result.append(score)

        return result
