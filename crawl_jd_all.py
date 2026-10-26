"""爬取京东帮助中心全部条款。

流程：先遍历所有分类和分页，收集文章链接；再逐篇抓取正文。
已下载的文章会跳过，中断后重跑可以断点续爬。

跑法：
    python crawl_jd_all.py
"""

import glob
import os
import re
import time

import requests
from bs4 import BeautifulSoup

BASE = "https://help.jd.com/user/issue"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
OUT_DIR = os.path.join("docs", "电商客服")
DELAY = 0.5


def fetch(url):
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    return resp.text


def collect_articles():
    """遍历所有分类和分页，返回去重后的文章 URL 列表。"""
    index_html = fetch(f"{BASE}.html")
    category_ids = sorted(
        set(re.findall(r"/user/issue/list-(\d+)\.html", index_html))
    )
    print(f"分类数：{len(category_ids)}")

    urls = set()
    for category_id in category_ids:
        page = 1
        while page <= 10:
            suffix = "" if page == 1 else f"?page={page}"
            try:
                html = fetch(f"{BASE}/list-{category_id}.html{suffix}")
            except Exception:
                break

            found = set(re.findall(r"/user/issue/(\d+-\d+)\.html", html))
            new_urls = {f"{BASE}/{item}.html" for item in found} - urls
            if not new_urls:
                break
            urls |= new_urls
            page += 1
            time.sleep(0.2)

        print(f"  分类 {category_id} 完成，累计 {len(urls)} 篇")

    return sorted(urls)


def clean_text(text):
    """去掉帮助页底部的投票和推荐问题。"""
    for marker in ("这条帮助是否解决了您的问题", "猜你感兴趣的问题"):
        cut = text.find(marker)
        if cut != -1:
            text = text[:cut].rstrip()
    return text.replace("下载该帮助文档", "")


def save_article(url):
    """抓取一篇并保存，已存在则跳过。"""
    article_id = url.rstrip(".html").split("/")[-1]
    if glob.glob(os.path.join(OUT_DIR, f"{article_id}_*.md")):
        return "skip"

    soup = BeautifulSoup(fetch(url), "html.parser")
    node = soup.select_one("div.content")
    if node is None:
        return "empty"

    text = clean_text(node.get_text("\n", strip=True))
    if len(text) < 100:
        return "empty"

    lines = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip(">").strip()
        if cleaned:
            lines.append(cleaned)
    title = lines[0] if lines else article_id
    title = re.sub(r'[\\/:*?"<>|]', "_", title)[:40]

    path = os.path.join(OUT_DIR, f"{article_id}_{title}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{text}\n")
    return "saved"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    urls = collect_articles()
    print(f"\n共发现 {len(urls)} 篇文章，开始逐篇抓取\n")

    saved = skipped = failed = 0
    for index, url in enumerate(urls, start=1):
        try:
            result = save_article(url)
        except Exception:
            failed += 1
            continue

        if result == "saved":
            saved += 1
            time.sleep(DELAY)
        elif result == "skip":
            skipped += 1
        else:
            failed += 1

        if index % 50 == 0:
            print(f"  进度 {index}/{len(urls)}：新存 {saved}，跳过 {skipped}，失败 {failed}")

    print(f"\n完成：新存 {saved}，跳过 {skipped}，失败 {failed}")
    print(f"目录：{OUT_DIR}")


if __name__ == "__main__":
    main()
