"""Shared HTTP(S) URL checks (SSRF mitigation) for httpx and headless browser."""

import ipaddress
import socket
from urllib.parse import urlparse


def assert_public_http_url(url: str) -> None:
    """Only http(s); hostname must resolve to at least one public routable IP."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http and https URLs are allowed.")
    host = parsed.hostname
    if not host:
        raise ValueError("Invalid URL.")
    hl = host.lower()
    if hl in ("localhost", "0.0.0.0") or hl.endswith(".local"):
        raise ValueError("Host not allowed.")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve host: {e}") from e
    checked_any = False
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        checked_any = True
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise ValueError("URL must resolve to a public address.")
    if not checked_any:
        raise ValueError("Could not validate host address.")
