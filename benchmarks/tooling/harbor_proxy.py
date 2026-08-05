"""Render Harbor's transparent egress configuration with an upstream proxy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml


def _proxy_node(proxy_url: str) -> dict[str, object]:
    parsed = urlsplit(proxy_url)
    if parsed.scheme not in {"http", "socks5", "socks5h"}:
        raise ValueError("upstream proxy must use http://, socks5://, or socks5h://")
    if parsed.hostname is None:
        raise ValueError("upstream proxy URL must include a hostname")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError(
            "upstream proxy URL must not include a path, query, or fragment"
        )

    default_port = 80 if parsed.scheme == "http" else 1080
    port = parsed.port or default_port
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    connector: dict[str, object] = {
        "type": "http" if parsed.scheme == "http" else "socks5"
    }
    if parsed.username is not None:
        connector["auth"] = {
            "username": unquote(parsed.username),
            "password": unquote(parsed.password or ""),
        }

    return {
        "name": "upstream-proxy",
        "addr": f"{host}:{port}",
        "connector": connector,
        "dialer": {"type": "tcp"},
    }


def render_config(proxy_url: str) -> dict[str, object]:
    """Return a GOST config that preserves Harbor's allowlist and chains egress."""
    return {
        "services": [
            {
                "name": "transparent-egress",
                "addr": ":12345",
                "bypass": "allowlist",
                "sockopts": {"mark": 114514},
                "handler": {
                    "type": "red",
                    "chain": "upstream-proxy",
                    "metadata": {
                        "sniffing": True,
                        "sniffing.timeout": "5s",
                        "sniffing.fallback": True,
                    },
                },
                "listener": {"type": "red"},
            }
        ],
        "chains": [
            {
                "name": "upstream-proxy",
                "hops": [
                    {
                        "name": "upstream-proxy",
                        "bypass": "direct-private",
                        "sockopts": {"mark": 114514},
                        "nodes": [_proxy_node(proxy_url)],
                    }
                ],
            }
        ],
        "bypasses": [
            {
                "name": "allowlist",
                "whitelist": True,
                "reload": "1s",
                "file": {"path": "/opt/egress-sidecar/allowlist.txt"},
            },
            {
                "name": "direct-private",
                "matchers": [
                    "127.0.0.0/8",
                    "10.0.0.0/8",
                    "172.16.0.0/12",
                    "192.168.0.0/16",
                    "169.254.0.0/16",
                    "::1/128",
                    "fc00::/7",
                    "fe80::/10",
                ],
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proxy_url = os.environ.get("JACOBIAN_EVAL_UPSTREAM_PROXY", "")
    if not proxy_url:
        parser.error("JACOBIAN_EVAL_UPSTREAM_PROXY is required")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(render_config(proxy_url), sort_keys=False))
    args.output.chmod(0o600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
