"""重排序：把问题和候选段落拼在一起打分。

向量检索是双塔结构，问题和文档分别编码成向量再比距离，速度快但精度有限。
重排序模型是交叉编码器，把问题和文档一起送进模型，直接判断这段能不能回答问题，
准得多，但慢得多。

所以标准做法是两阶段：先用检索粗召回一批候选，再用重排序精排出前几条。
"""

import warnings

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "BAAI/bge-reranker-base"

warnings.filterwarnings("ignore", message=".*resume_download.*")

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        print("正在加载重排序模型（第一次会慢一点）...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, local_files_only=True
        )
        _model.eval()
    return _tokenizer, _model


def rerank(question, candidates, top_k):
    """candidates 是 [(分数, 文本, 来源), ...]，返回重排后的前 top_k 条。"""
    if not candidates:
        return []

    tokenizer, model = _load()
    pairs = [(question, text) for _, text, _ in candidates]
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**inputs).logits.view(-1)
    # 模型输出的是 logits，过一层 sigmoid 变成 0 到 1 的相关度
    scores = torch.sigmoid(logits).tolist()

    ranked = sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)
    return [(float(score), text, source) for score, (_, text, source) in ranked[:top_k]]
