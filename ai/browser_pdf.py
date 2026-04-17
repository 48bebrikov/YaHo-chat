"""Headless Chromium: full-page PDF capture for multimodal Gemini (charts, layout)."""

from __future__ import annotations

import logging

from ai.url_safety import assert_public_http_url

logger = logging.getLogger(__name__)

# Inline PDF size cap (Gemini / request limits); scale down if exceeded.
_MAX_PDF_BYTES = 4_500_000


def _scroll_lazy(page) -> None:
    """Scroll to bottom in steps so lazy-loaded images/charts can render."""
    prev = -1
    for _ in range(50):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(350)
        h = page.evaluate("document.body.scrollHeight")
        if h == prev:
            break
        prev = h
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(200)


def capture_url_as_pdf(url: str) -> tuple[bytes | None, str | None]:
    """
    Returns (pdf_bytes, error_message). error_message is None on success.
    """
    url = (url or "").strip()
    if not url:
        return None, "Error: empty URL."
    try:
        assert_public_http_url(url)
    except ValueError as e:
        return None, f"Cannot open URL: {e}"

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None, "Playwright is not installed. Run: pip install playwright && playwright install chromium"

    pdf_bytes: bytes | None = None

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        try:
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 720},
                locale="ru-RU",
                extra_http_headers={
                    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                },
            )
            page = context.new_page()

            def _guard(req):
                u = req.url
                if not u.startswith(("http://", "https://")):
                    return
                try:
                    assert_public_http_url(u)
                except ValueError:
                    req.abort()

            page.on("request", _guard)

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=90_000)
            except Exception as e:
                return None, f"Navigation failed: {e}"

            try:
                _scroll_lazy(page)
            except Exception as ex:
                logger.debug("Scroll helper failed (non-fatal): %s", ex)

            page.wait_for_timeout(800)

            for scale in (1.0, 0.85, 0.7, 0.55):
                try:
                    pdf_bytes = page.pdf(
                        full_page=True,
                        print_background=True,
                        scale=scale,
                    )
                except Exception as e:
                    return None, f"PDF export failed: {e}"
                if pdf_bytes and len(pdf_bytes) <= _MAX_PDF_BYTES:
                    break
        finally:
            browser.close()

    if not pdf_bytes:
        return None, "Empty PDF."
    if len(pdf_bytes) > _MAX_PDF_BYTES:
        return None, "Page too large for PDF upload; try browse_url (text) or a simpler page."
    return pdf_bytes, None
