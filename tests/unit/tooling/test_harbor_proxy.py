from __future__ import annotations

import pytest
from benchmarks.tooling.harbor_proxy import render_config


def test_render_config_chains_transparent_egress_through_http_proxy() -> None:
    config = render_config("http://docker-host:7890")

    transparent_service = config["services"][0]
    explicit_service = config["services"][1]
    hop = config["chains"][0]["hops"][0]
    node = hop["nodes"][0]
    assert transparent_service["handler"]["chain"] == "upstream-proxy"
    assert transparent_service["bypass"] == "allowlist"
    assert transparent_service["sockopts"] == {"mark": 114514}
    assert explicit_service == {
        "name": "explicit-egress",
        "addr": "127.0.0.1:12346",
        "bypass": "allowlist",
        "handler": {"type": "http", "chain": "upstream-proxy"},
        "listener": {"type": "tcp"},
    }
    assert hop["bypass"] == "direct-private"
    assert hop["sockopts"] == {"mark": 114514}
    assert node["addr"] == "docker-host:7890"
    assert node["connector"] == {"type": "http"}
    assert config["bypasses"][1]["matchers"] == [
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    ]


def test_render_config_rejects_unsupported_proxy_scheme() -> None:
    with pytest.raises(ValueError, match="upstream proxy must use"):
        render_config("https://docker-host:7890")
