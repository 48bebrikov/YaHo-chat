import re
from html.parser import HTMLParser

import httpx
from ddgs import DDGS
from database.sqlite_db import db_session, NewsCache
from ai.url_safety import assert_public_http_url

_MAX_FETCH_BYTES = 1_500_000
_MAX_BROWSE_OUTPUT_CHARS = 80_000


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._chunks = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg", "template"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg", "template"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._chunks.append(data)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


def _extract_charset(content_type: str | None) -> str | None:
    if not content_type:
        return None
    m = re.search(r"charset=([\w-]+)", content_type, re.I)
    return m.group(1).strip() if m else None


def browse_url(url: str) -> str:
    """Loads a public web page by URL and returns the main text (HTML tags and scripts removed).
    Use when the friend sent a link, or you need details from a specific page (article, docs, etc.)."""
    url = (url or "").strip()
    if not url:
        return "Error: empty URL."
    try:
        assert_public_http_url(url)
    except ValueError as e:
        return f"Cannot open URL: {e}"

    def _hook(request: httpx.Request) -> None:
        assert_public_http_url(str(request.url))

    try:
        with httpx.Client(
            timeout=httpx.Timeout(20.0, connect=10.0),
            follow_redirects=True,
            max_redirects=5,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; YaHoBot/1.0; +https://t.me/)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            event_hooks={"request": [_hook]},
        ) as client:
            r = client.get(url)
        r.raise_for_status()
        body = r.content[:_MAX_FETCH_BYTES]
        ct = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
        charset = _extract_charset(r.headers.get("content-type"))

        if ct and not (ct.startswith("text/") or ct in ("application/json", "application/xml", "application/xhtml+xml")):
            return f"Error: Unsupported content type '{ct}'. I can only read text/html/json/xml."

        if ct in ("text/plain", "application/json", "application/xml", "text/xml"):
            try:
                text = body.decode(charset or "utf-8", errors="replace")
            except LookupError:
                text = body.decode("utf-8", errors="replace")
        else:
            try:
                raw = body.decode(charset or "utf-8", errors="replace")
            except LookupError:
                raw = body.decode("utf-8", errors="replace")
            parser = _HTMLTextExtractor()
            try:
                parser.feed(raw)
            except Exception:
                pass
            text = parser.get_text() or raw

        if len(text) > _MAX_BROWSE_OUTPUT_CHARS:
            text = text[:_MAX_BROWSE_OUTPUT_CHARS] + "\n\n[…truncated…]"
        final_url = str(r.url)
        return f"URL (final): {final_url}\nStatus: {r.status_code}\n\n{text}"
    except httpx.HTTPStatusError as e:
        return f"HTTP error opening URL: {e.response.status_code}"
    except httpx.RequestError as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"Error reading URL: {e}"

# If duckduckgo.com is unreachable from your server, set DDGS_PROXY (e.g. http://host:3128)
# in the container environment — DDGS reads it automatically.


def _ddgs_text(query: str, max_results: int = 3):
    """
    Text search: metasearch (auto) first, then engines that often work when DDG HTML is blocked.
    """
    timeout = 30
    attempts = (
        {"backend": "auto", "region": "ru-ru"},
        {"backend": "yandex,wikipedia", "region": "ru-ru"},
        {"backend": "wikipedia", "region": "en-us"},
    )
    last_err = None
    for kwargs in attempts:
        try:
            with DDGS(timeout=timeout) as ddgs:
                results = ddgs.text(query, max_results=max_results, **kwargs)
            if results:
                return results
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    return []


def search_internet(query: str) -> str:
    """Searches the internet for general information or memes."""
    try:
        results = _ddgs_text(query, max_results=3)
        if not results:
            return "No results found."

        snippets = []
        for r in results:
            snippets.append(
                f"Title: {r.get('title', '')}\nSnippet: {r.get('body', '')}\nLink: {r.get('href', '')}"
            )

        return "\n\n".join(snippets)
    except Exception as e:
        return f"Error searching internet: {e}"

def search_youtube(query: str) -> str:
    """Searches YouTube for videos."""
    try:
        with DDGS(timeout=30) as ddgs:
            results = ddgs.videos(query, max_results=3)
        if not results:
            return "No videos found."

        snippets = []
        for r in results:
            snippets.append(f"Video Title: {r.get('title')}\nDescription: {r.get('description')}\nURL: {r.get('content')}")

        return "\n\n".join(snippets)
    except Exception as e:
        return f"Error searching youtube: {e}"

def search_saved_news(query: str) -> str:
    """Searches the local database of recently saved posts from Telegram channels."""
    try:
        with db_session() as db:
            # Simple LIKE search. For better search, we could use FTS or Qdrant for news too.
            # But for now, basic LIKE is sufficient or just return the latest N news if query is empty.
            if not query or query.lower() in ["latest", "news", "новости"]:
                news = db.query(NewsCache).order_by(NewsCache.date_added.desc()).limit(5).all()
            else:
                search_pattern = f"%{query}%"
                news = db.query(NewsCache).filter(NewsCache.text.ilike(search_pattern)).order_by(NewsCache.date_added.desc()).limit(5).all()
            
            if not news:
                return "No saved news found."
                
            snippets = []
            for n in news:
                snippets.append(f"[{n.date_added.strftime('%Y-%m-%d %H:%M')}] News from channel {n.channel_id}:\n{n.text}")
                
            return "\n\n".join(snippets)
    except Exception as e:
        return f"Error searching news: {e}"


def browse_url_visual(url: str) -> str:
    """Renders the page in headless Chromium (real browser), scrolls to load lazy content, exports full page as PDF for vision (charts, screenshots, layout). Slower than browse_url. Actual PDF is attached by the runtime — do not rely on this return value."""
    return ""
