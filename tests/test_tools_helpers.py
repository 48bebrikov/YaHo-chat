from ai.tools import _HTMLTextExtractor, _extract_charset


def test_extract_charset_from_content_type():
    assert _extract_charset("text/html; charset=utf-8") == "utf-8"
    assert _extract_charset('application/xml; Charset=ISO-8859-1') == "ISO-8859-1"
    assert _extract_charset(None) is None
    assert _extract_charset("text/html") is None


def test_html_text_extractor_skips_script_and_keeps_body():
    p = _HTMLTextExtractor()
    p.feed("<script>evil()</script><p>Hello</p><style>.x{}</style><span>World</span>")
    text = p.get_text()
    assert "evil" not in text
    assert "Hello" in text
    assert "World" in text


def test_browse_url_empty_returns_error():
    from ai.tools import browse_url

    assert "empty" in browse_url("").lower()


def test_browse_url_blocks_bad_scheme_before_fetch():
    from ai.tools import browse_url

    out = browse_url("file:///etc/passwd")
    assert "Cannot open" in out or "http" in out.lower()
