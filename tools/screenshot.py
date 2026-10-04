"""Render an HTML file to PNG with headless Chromium (Playwright). Usage: screenshot.py in.html out.png [width]"""
import sys
from playwright.sync_api import sync_playwright
src, out = sys.argv[1], sys.argv[2]
w = int(sys.argv[3]) if len(sys.argv) > 3 else 1400
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": w, "height": 200}, device_scale_factor=2)
    pg.goto("file://" + __import__("os").path.abspath(src))
    pg.screenshot(path=out, full_page=True)
    b.close()
