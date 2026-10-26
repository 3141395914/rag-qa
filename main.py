"""命令行入口。

建索引：python main.py index
提问题：python main.py ask "你的问题"
"""

import os
import sys

from generator import answer
from retriever import build_index, search

DOCS_DIR = "docs"


def load_docs():
    """递归读取 docs 目录下的所有文件，子目录也会被扫到。"""
    docs = []
    for root, _, files in os.walk(DOCS_DIR):
        for name in sorted(files):
            path = os.path.join(root, name)
            rel_path = os.path.relpath(path, DOCS_DIR)
            with open(path, encoding="utf-8") as f:
                docs.append((rel_path, f.read()))
    return docs


def main():
    if len(sys.argv) < 2:
        print('用法：python main.py index')
        print('      python main.py ask "问题"')
        return

    command = sys.argv[1]

    if command == "index":
        docs = load_docs()
        if not docs:
            print(f"{DOCS_DIR} 目录里没有文档")
            return
        print(f"读取到 {len(docs)} 个文档")
        build_index(docs)
        return

    if command == "ask":
        question = " ".join(sys.argv[2:]).strip()
        if not question:
            print("请把问题写在命令后面")
            return

        hits = search(question, rerank=True)
        print("检索到的资料：")
        for i, (score, text, source) in enumerate(hits):
            print(f"  [{i + 1}] 相似度 {score:.3f}  {source}")
        print("\n回答：")
        print(answer(question, hits))
        return

    print("不认识的命令：", command)


if __name__ == "__main__":
    main()
