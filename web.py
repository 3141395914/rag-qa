"""网页版本：用 FastAPI 提供接口，前端放在 static/index.html。

跑法：
    python -m uvicorn web:app --port 8000

打开 http://127.0.0.1:8000
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from generator import answer
from retriever import search

app = FastAPI(title="电商客服知识库")


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(item: Question):
    """接收问题，返回答案和引用来源。"""
    hits = search(item.question, rerank=True)
    reply = answer(item.question, hits)
    sources = [
        {
            "score": round(score, 3),
            "source": source,
            "text": text[:300],
        }
        for score, text, source in hits
    ]
    return {"answer": reply, "sources": sources}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
