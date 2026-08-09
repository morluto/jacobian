"""Behavioral tests for the shared provider-spike utility helpers."""

from __future__ import annotations

from benchmarks.tooling.spike_utils import (
    canonical_json,
    sha256_bytes,
)


def test_sha256_bytes_uses_prefixed_sha256() -> None:
    assert sha256_bytes(b"provider-spike-fixture") == (
        "sha256:288978a7308f68a579a0867384dc2ea237e5190a353f5619dfcd10b2d933ec21"
    )


def test_canonical_json_produces_stable_ascii_wire_bytes() -> None:
    assert canonical_json({"z": {"key": "\u00e9"}, "a": [3, 2, 1]}) == (
        b'{"a":[3,2,1],"z":{"key":"\\u00e9"}}\n'
    )
