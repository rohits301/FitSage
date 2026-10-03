"""Record docs/demo.gif and docs/demo.mp4 from the REAL running app (no mock-ups).

    pip install playwright && playwright install chromium     # one-off, dev only
    python scripts/record_demo.py

It starts uvicorn, drives the page with a headless browser, types real questions,
and stitches screenshots of what the app actually returned. If the retriever or the
corpus changes, re-run it so the demo never drifts from the code.
"""
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PORT = 8765
URL = f"http://127.0.0.1:{PORT}"
W, H = 1040, 720

SCENES = [
    # (how to enter the question, text)
    ("chip", "What is the daily vitamin D recommendation for grown-ups, in IU?"),
    ("type", "Is it safe to take omega-3 supplements with warfarin?"),
    ("type", "Which running shoes should I buy?"),
]


def wait_up():
    for _ in range(60):
        try:
            urllib.request.urlopen(f"{URL}/api/health", timeout=1)
            return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("server did not start")


def main() -> None:
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT), "--log-level", "warning"],
        cwd=ROOT,
    )
    frames: list[tuple[Image.Image, int]] = []
    try:
        wait_up()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": W, "height": H})
            page.goto(URL, wait_until="domcontentloaded")
            page.wait_for_timeout(1200)

            def snap(ms: int) -> None:
                tmp = Path(tempfile.mkdtemp()) / "f.png"
                page.screenshot(path=str(tmp))
                frames.append((Image.open(tmp).convert("RGB"), ms))

            snap(1800)
            for how, text in SCENES:
                page.fill("#question", "")
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(400)
                if how == "chip":
                    page.click(f'[data-question="{text}"]')
                    page.wait_for_timeout(300)
                    snap(900)
                else:
                    page.focus("#question")
                    for i in range(0, len(text), 6):
                        page.keyboard.type(text[i:i + 6], delay=0)
                        page.wait_for_timeout(60)
                        snap(110)
                    snap(500)
                page.click("#submit")
                page.wait_for_function("document.querySelector('#submit').disabled === false")
                page.wait_for_timeout(900)  # smooth-scroll to the result
                snap(3200)
                page.mouse.wheel(0, 380)
                page.wait_for_timeout(500)
                snap(3200)
            snap(800)
            browser.close()
    finally:
        server.terminate()

    DOCS.mkdir(exist_ok=True)
    gw = 880
    small = [(im.resize((gw, round(im.height * gw / im.width)), Image.LANCZOS), ms) for im, ms in frames]
    pal = [im.quantize(colors=96, method=Image.MEDIANCUT, dither=Image.NONE) for im, _ in small]
    pal[0].save(
        DOCS / "demo.gif", save_all=True, append_images=pal[1:], duration=[ms for _, ms in small],
        loop=0, optimize=True, disposal=1,
    )

    tmpdir = Path(tempfile.mkdtemp())
    lines = []
    for i, (im, ms) in enumerate(frames):
        f = tmpdir / f"{i:03d}.png"
        im.save(f)
        lines += [f"file '{f}'", f"duration {ms / 1000:.3f}"]
    lines.append(f"file '{tmpdir / f'{len(frames) - 1:03d}.png'}'")
    (tmpdir / "list.txt").write_text("\n".join(lines))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmpdir / "list.txt"),
         "-vf", "fps=25,scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-c:v", "libx264", "-crf", "26",
         "-movflags", "+faststart", str(DOCS / "demo.mp4")],
        check=True,
    )
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("wrote", DOCS / "demo.gif", f"({(DOCS / 'demo.gif').stat().st_size // 1024} KB)", "and", DOCS / "demo.mp4")


if __name__ == "__main__":
    main()
