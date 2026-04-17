from unittest.mock import patch

import pytest

from ai.url_safety import assert_public_http_url


@patch("ai.url_safety.socket.getaddrinfo")
def test_accepts_public_resolved_ip(mock_gai):
    mock_gai.return_value = [(None, None, None, None, ("8.8.8.8", 0))]
    assert_public_http_url("https://example.com/path")


@patch("ai.url_safety.socket.getaddrinfo")
def test_rejects_resolved_private_ip(mock_gai):
    mock_gai.return_value = [(None, None, None, None, ("10.0.0.1", 0))]
    with pytest.raises(ValueError, match="public"):
        assert_public_http_url("https://example.com/")


def test_rejects_non_http_scheme():
    with pytest.raises(ValueError, match="http"):
        assert_public_http_url("ftp://example.com/")


def test_rejects_empty_host():
    with pytest.raises(ValueError):
        assert_public_http_url("http:///no-host")


def test_rejects_localhost_name():
    with pytest.raises(ValueError):
        assert_public_http_url("http://localhost/foo")


def test_rejects_dot_local():
    with pytest.raises(ValueError):
        assert_public_http_url("http://machine.local/")
