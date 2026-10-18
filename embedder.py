"""加载本地向量模型，把文字转成向量。"""

import warnings

import torch
from transformers import AutoModel, AutoTokenizer

from config import MODEL_NAME, QUERY_PREFIX

warnings.filterwarnings("ignore", message=".*resume_download.*")

_tokenizer = None
_model = None


def _load():
    """第一次调用时加载模型，之后复用。"""
    global _tokenizer, _model
    if _model is None:
        print("正在加载向量模型（第一次会慢一点）...")
        _tokenizer = _load_any(AutoTokenizer, MODEL_NAME)
        _model = _load_any(AutoModel, MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def _load_any(cls, name):
    """优先读本地缓存；本地没有才联网下载。

    这样模型下载过一次之后，每次启动都不用再访问网络。
    """
    try:
        return cls.from_pretrained(name, local_files_only=True)
    except OSError:
        print("本地没有缓存，开始联网下载模型...")
        return cls.from_pretrained(name)


def encode(texts):
    """把一批文本转成归一化后的向量。"""
    tokenizer, model = _load()
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # BGE 系列模型取 [CLS] 位置的向量作为整句表示
    vectors = outputs.last_hidden_state[:, 0]
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
    return vectors.numpy()


def encode_query(question):
    """问题向量。BGE 建议检索问题加一个指令前缀。"""
    return encode([QUERY_PREFIX + question])[0]
