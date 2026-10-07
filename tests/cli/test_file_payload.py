"""The CLI bounds file allocation before strict JSON admission."""

from __future__ import annotations

import json
import subprocess
import sys
import tracemalloc
from pathlib import Path

import pytest

from jacobian.canonical import CanonicalizationError, CanonicalLimits
from jacobian.cli import run_operation


def test_cli_file_input_allocation_stays_within_the_parser_budget(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "oversized.json"
    limit = CanonicalLimits().max_input_bytes
    with payload.open("wb") as stream:
        stream.truncate(limit * 4)
    tracemalloc.start()
    try:
        with pytest.raises(CanonicalizationError, match="size limit"):
            run_operation("integer.compute.extended_gcd", file=payload)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < limit * 2, "file reader allocated beyond the parser input budget"


def test_real_cli_rejects_oversized_file_as_invalid_argument(tmp_path: Path) -> None:
    payload = tmp_path / "oversized.json"
    with payload.open("wb") as stream:
        stream.truncate(CanonicalLimits().max_input_bytes + 1)
    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "jacobian"),
            "run",
            "integer.compute.extended_gcd",
            "--file",
            str(payload),
        ],
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 1
    assert result.stdout == b""
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "INVALID_ARGUMENT"
    assert "size limit" in error["message"]


def test_cli_file_payload_computes_the_exact_result(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text('{"left":"84","right":"30"}', encoding="utf-8")
    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "jacobian"),
            "run",
            "integer.compute.extended_gcd",
            "--file",
            str(payload),
        ],
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    output = json.loads(result.stdout)["output"]
    assert int(output["gcd"]) == 6
    assert (
        84 * int(output["left_coefficient"]) + 30 * int(output["right_coefficient"])
        == 6
    )
    assert result.stderr == b""
