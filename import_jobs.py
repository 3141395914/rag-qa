"""把爬虫抓到的岗位导入 RAG 文档库。

跑法：
    python import_jobs.py D:\\codexjob\\web-extract-agent\\cache

脚本读缓存目录里所有 JSON，把岗位汇总成一份 Markdown 文档，
每个岗位一段，用 --- 分隔。之后重新建索引就能检索这些岗位了。
"""

import json
import os
import sys

OUT_PATH = os.path.join("docs", "招聘岗位.md")


def load_jobs(cache_dir):
    """读缓存目录里的所有岗位。"""
    jobs = []
    if not os.path.isdir(cache_dir):
        print("目录不存在：", cache_dir)
        return jobs

    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(cache_dir, name), encoding="utf-8") as f:
            data = json.load(f)
        for job in data.get("jobs", []):
            if job.get("job_title"):
                jobs.append(job)
    return jobs


def dedupe(jobs):
    """同一个岗位可能在多个页面出现，按公司加岗位名去重。"""
    seen = set()
    result = []
    for job in jobs:
        key = (job.get("company"), job.get("job_title"))
        if key in seen:
            continue
        seen.add(key)
        result.append(job)
    return result


def format_job(job):
    """把一个岗位写成一段话，方便检索时命中。"""
    return "\n".join(
        [
            f"岗位：{job.get('job_title')}",
            f"公司：{job.get('company') or '未写明'}",
            f"薪资：{job.get('salary') or '未写明'}",
            f"学历要求：{job.get('education') or '未写明'}",
            f"工作地点：{job.get('location') or '未写明'}",
            f"来源：{job.get('url') or ''}",
        ]
    )


def main():
    cache_dir = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join("..", "web-extract-agent", "cache")
    )

    jobs = load_jobs(cache_dir)
    unique = dedupe(jobs)

    text = "\n\n---\n\n".join(format_job(job) for job in unique)
    os.makedirs("docs", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"读取 {len(jobs)} 条，去重后 {len(unique)} 个岗位")
    print("已写入", OUT_PATH)


if __name__ == "__main__":
    main()
