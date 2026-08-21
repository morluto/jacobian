"""Behavioral tests for the shared provider-spike utility helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.tooling.spike_utils import (
    canonical_json,
    owned_fixture_path,
    sha256_bytes,
)

ROOT = Path(__file__).resolve().parents[2]


def test_sha256_bytes_uses_prefixed_sha256() -> None:
    assert sha256_bytes(b"provider-spike-fixture") == (
        "sha256:288978a7308f68a579a0867384dc2ea237e5190a353f5619dfcd10b2d933ec21"
    )


def test_canonical_json_produces_stable_ascii_wire_bytes() -> None:
    assert canonical_json({"z": {"key": "\u00e9"}, "a": [3, 2, 1]}) == (
        b'{"a":[3,2,1],"z":{"key":"\\u00e9"}}\n'
    )


@pytest.mark.parametrize("provider", ["cddlib", "gudhi", "regina"])
def test_checked_in_pin_binds_the_moved_spike_source(provider: str) -> None:
    source = ROOT / "benchmarks/tooling/providers" / f"{provider}.py"
    pin_path = ROOT / "tests/fixtures/providers" / provider / "pin.json"
    pin = json.loads(pin_path.read_text(encoding="utf-8"))

    assert pin["adapter_source_sha256"] == sha256_bytes(source.read_bytes())


def test_owned_fixture_path_falls_back_beside_a_copied_script(
    tmp_path: Path,
) -> None:
    script = tmp_path / "spike.py"
    script.touch()

    assert owned_fixture_path(script.as_posix(), "missing/pin.json", "pin.json") == (
        tmp_path / "pin.json"
    )


def test_owned_fixture_path_handles_the_shallow_container_module_path() -> None:
    assert owned_fixture_path(
        "/app/spike.py", "tests/fixtures/providers/cddlib", "pin.json"
    ) == Path("/app/pin.json")
