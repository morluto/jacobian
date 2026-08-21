"""Canonical MCP server-surface probing and digest construction.

Surface probing owns only the live MCP/server contract and its digest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _json_digest(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    )


def surface_snapshot_digest(surface: Mapping[str, Any]) -> str:
    """Return a stable digest of the observed server surface."""
    return _json_digest(surface)


__all__ = [
    "surface_snapshot_digest",
]
