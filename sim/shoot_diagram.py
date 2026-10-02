"""
shoot_diagram.py -- Screenshot the architecture diagram HTML at 2x device
scale (300dpi-equivalent) for embedding as a ReportLab Image() flowable.

Per SKILL.md "Diagram Generation Strategy" (Report route): diagrams are
rendered via Playwright page.screenshot at 2x; the document-level PDF is
still produced by ReportLab (vector).
"""
import asyncio
import os

from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "fig1_architecture.html")
OUT = os.path.join(HERE, "figs", "fig1_architecture.png")


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(
            viewport={"width": 1080, "height": 900},
            device_scale_factor=2,
        )
        await page.goto("file://" + HTML)
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(400)
        el = await page.query_selector(".canvas")
        await el.screenshot(path=OUT)
        await browser.close()
    print("wrote", OUT)


if __name__ == "__main__":
    asyncio.run(main())
