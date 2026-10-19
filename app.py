"""网页界面：在浏览器里提问。

跑法：python app.py
启动后打开 http://127.0.0.1:7860
"""

import gradio as gr

from generator import answer
from retriever import search


def ask(question):
    """检索、生成，返回答案和参考来源。"""
    if not question.strip():
        return "请输入问题", ""

    hits = search(question, rerank=True)
    reply = answer(question, hits)
    sources = "\n\n".join(
        f"**[{i + 1}] {source}**（相关度 {score:.3f}）\n\n{text}"
        for i, (score, text, source) in enumerate(hits)
    )
    return reply, sources


with gr.Blocks(title="文档知识库问答") as demo:
    gr.Markdown("# 文档知识库问答")
    gr.Markdown("先检索本地文档，再让模型根据资料回答，答案后面标出引用来源。")

    question = gr.Textbox(
        label="问题",
        placeholder="例如：这个项目为什么要做结果校验？",
        lines=2,
    )
    submit = gr.Button("提问", variant="primary")

    reply = gr.Textbox(label="回答", lines=8)
    gr.Markdown("### 参考来源")
    sources = gr.Markdown()

    submit.click(ask, inputs=question, outputs=[reply, sources])
    question.submit(ask, inputs=question, outputs=[reply, sources])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
