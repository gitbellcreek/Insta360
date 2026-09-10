"""Browser smoke test for the web app in docs/.

    pip install playwright && playwright install chromium
    python tests/web_smoke.py /path/to/folder_with_insp_files [width]

Serves docs/ on a local port, opens it in headless Chromium, loads every
.insp in the folder through the folder input, presses "Stitch all" and
reports the alignment result per file.  Set CHROMIUM=/path/to/chromium to
use a specific browser binary.
"""
import functools
import json
import os
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parents[1] / "docs"


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    width = sys.argv[2] if len(sys.argv) > 2 else "2048"
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(DOCS))
    handler.log_message = lambda *a, **k: None
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    with sync_playwright() as p:
        kw = {"args": ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-proxy-server"]}
        if os.environ.get("CHROMIUM"):
            kw["executable_path"] = os.environ["CHROMIUM"]
        browser = p.chromium.launch(**kw)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.on("pageerror", lambda e: print("pageerror:", e))
        page.on("console", lambda m: print("console:", m.text) if m.type == "error" and "favicon" not in m.text else None)
        page.goto(f"http://127.0.0.1:{port}/index.html")
        page.wait_for_function("window.app && window.app.stitcher")
        page.set_input_files("#folderInput", folder)
        page.wait_for_function("window.app.entries.length > 0")
        page.select_option("#width", width)
        page.click("#btnStitchAll")
        t = time.time()
        last = ""
        while time.time() - t < 3600:
            s = page.text_content("#status")
            if s != last:
                print(f"[{time.time() - t:6.1f}s] {s}")
                last = s
            if time.time() - t > 3 and page.evaluate("!window.app.busy && window.app.queue.length === 0"):
                break
            time.sleep(0.5)
        res = page.evaluate("window.app.entries.map(e => ({name: e.name, ok: !!e.blob, alignment: e.info && e.info.alignment, levelled: e.info && e.info.levelled, seconds: e.info && e.info.seconds}))")
        print(json.dumps(res, indent=1))
        browser.close()
        return 0 if all(r["ok"] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
