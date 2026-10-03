"""Check that every corpus passage still appears in the live source page.

    pip install playwright && playwright install chromium     # dev only
    python scripts/verify_corpus.py

For each source URL the page is rendered in a headless browser and its visible text is
whitespace-normalised (bracketed citation numbers such as [12,13] removed). Each
passage must then appear in that text as an exact substring. Table and list passages
are stored exactly as the page's text renders them.

Some sites throw bot checks at automated browsers; those pages are reported as
UNREACHABLE rather than as failures, and the script exits non-zero if anything is
either unreachable or not found.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CITATION = re.compile(r"\s?\[\d+(?:[,–-]\d+)*\]")


def norm(text: str) -> str:
    return re.sub(r"[ \s]+", " ", text).replace("’", "'").strip()


def page_text(page, url: str, min_chars: int = 500) -> str:
    for _ in range(3):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(5000)
            text = norm(page.inner_text("body"))
        except Exception:
            continue
        if "Just a moment" not in page.title() and len(text) > min_chars:
            return CITATION.sub("", text)
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "app" / "data" / "corpus.json"))
    ap.add_argument("--min-chars", type=int, default=500, help="shorter pages are treated as blocked")
    args = ap.parse_args()
    passages = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    by_url = defaultdict(list)
    for p in passages:
        by_url[p["url"]].append(p)
    bad = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        for url, items in by_url.items():
            text = page_text(page, url, args.min_chars)
            if not text:
                print(f"UNREACHABLE  {url}")
                bad += len(items)
                continue
            for p in items:
                ok = norm(p["text"]) in text
                bad += not ok
                print(f"{'OK        ' if ok else 'NOT FOUND '}  {p['id']:16} {url}")
        browser.close()
    print(f"\n{len(passages) - bad}/{len(passages)} passages verified")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
