"""把检索到的资料和问题拼成提示词，交给 DeepSeek 回答。"""

import os

import requests
from dotenv import load_dotenv

from config import API_URL, MODEL, TIMEOUT

load_dotenv()


def build_prompt(question, hits):
    context = "\n\n".join(
        f"[{i + 1}] 来源：{source}\n{text}"
        for i, (score, text, source) in enumerate(hits)
    )
    return (
        "你是一个资料问答助手。请根据下面提供的资料回答问题。\n"
        "资料里如果有相关信息，请归纳总结后回答；只有资料完全没有相关内容时，"
        "才回答「资料里没有提到」，不要自己编。\n"
        "回答时用 [1] [2] 这样的编号标出依据来自哪段资料。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{question}"
    )


def answer(question, hits):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("没有找到 DEEPSEEK_API_KEY，请检查 .env 文件")

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": build_prompt(question, hits)}],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
