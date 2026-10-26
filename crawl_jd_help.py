"""爬取京东帮助中心的公开政策文档，整理成 RAG 知识库。

只抓帮助中心，不碰商品页；每篇之间停顿一秒，低频访问，用于个人学习。

跑法：
    python crawl_jd_help.py

输出到 docs/电商客服/，每个文件是一篇帮助文档。
"""

import os
import re
import time

import requests
from bs4 import BeautifulSoup

BASE = "https://help.jd.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# 购物指南、配送、支付、发票、售后等几个分类
CATEGORY_IDS = ["112", "113", "114", "115", "116", "117", "132", "133"]
MAX_PER_CATEGORY = 8
OUT_DIR = os.path.join("docs", "电商客服")


def fetch(url):
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    return resp.text


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seen = set()
    count = 0

    for category_id in CATEGORY_IDS:
        list_url = f"{BASE}/user/issue/list-{category_id}.html"
        try:
            html = fetch(list_url)
        except Exception as e:
            print(f"分类 {category_id} 抓取失败：{type(e).__name__}")
            continue

        links = sorted(
            set(re.findall(r'href="([^"]*?/user/issue/\d+-\d+\.html)"', html))
        )[:MAX_PER_CATEGORY]
        print(f"分类 {category_id}：发现 {len(links)} 篇")

        for link in links:
            url = "https:" + link if link.startswith("//") else link
            if url in seen:
                continue
            seen.add(url)

            try:
                soup = BeautifulSoup(fetch(url), "html.parser")
                node = soup.select_one("div.content")
                if node is None:
                    continue

                text = node.get_text("\n", strip=True)
                if len(text) < 100:
                    continue

                # 页面底部有"是否有帮助"投票和推荐问题，从标记处截断
                for marker in ("这条帮助是否解决了您的问题", "猜你感兴趣的问题"):
                    cut = text.find(marker)
                    if cut != -1:
                        text = text[:cut].rstrip()
                text = text.replace("下载该帮助文档", "")

                lines = []
                for line in text.splitlines():
                    cleaned = line.strip().lstrip(">").strip()
                    if cleaned:
                        lines.append(cleaned)
                title = lines[0] if lines else f"帮助文档{count + 1}"
                title = re.sub(r'[\\/:*?"<>|]', "_", title)[:40]

                count += 1
                path = os.path.join(OUT_DIR, f"{count:03d}_{title}.md")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"# {title}\n\n{text}\n")

                print(f"  保存 {title}")
                time.sleep(1)
            except Exception as e:
                print(f"  跳过一篇：{type(e).__name__}")

    print(f"\n共保存 {count} 篇到 {OUT_DIR}")


if __name__ == "__main__":
    main()
