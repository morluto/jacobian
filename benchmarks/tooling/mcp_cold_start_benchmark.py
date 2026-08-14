"""Measure ten isolated catalog-only MCP launches on the Linux runner."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, cast

_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.tooling.command_runner import (  # noqa: E402
    ToolCommandRequest,
    ToolCommandStatus,
    operator_environment,
    run_tool_command,
)

_LAUNCHES = 10
_MEDIAN_LIMIT_SECONDS = 2.0
_P95_LIMIT_SECONDS = 3.0


def _launch(state_dir: Path, *, audit: bool = False) -> dict[str, Any]:
    arguments = [
        "-m",
        "benchmarks.tooling.mcp_cold_start_probe",
        "--state-dir",
        str(state_dir.resolve(strict=True)),
    ]
    if audit:
        arguments.append("--audit")
    result = run_tool_command(
        ToolCommandRequest(
            executable=str(Path(sys.executable).absolute()),
            arguments=tuple(arguments),
            environment=operator_environment(),
            cwd=str(_ROOT),
            timeout_seconds=15.0,
            stdout_limit_bytes=16 * 1024,
            stderr_limit_bytes=16 * 1024,
        )
    )
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        raise RuntimeError(
            "cold MCP probe failed: " + result.stderr.decode("utf-8", errors="replace")
        )
    return cast(dict[str, Any], json.loads(result.stdout))


def run_benchmark(state_dir: Path) -> dict[str, object]:
    audit = _launch(state_dir, audit=True)
    launches = tuple(_launch(state_dir) for _ in range(_LAUNCHES))
    timings = sorted(float(item["elapsed_seconds"]) for item in launches)
    p95_index = math.ceil(0.95 * len(timings)) - 1
    median_seconds = statistics.median(timings)
    p95_seconds = timings[p95_index]
    forbidden = sorted(
        {
            module
            for item in (audit, *launches)
            for module in item["forbidden_startup_modules"]
        }
    )
    forbidden_calls = sorted(
        {
            call
            for item in (audit, *launches)
            for call in item["forbidden_startup_calls"]
        }
    )
    report: dict[str, object] = {
        "launches": _LAUNCHES,
        "median_seconds": median_seconds,
        "p95_seconds": p95_seconds,
        "forbidden_startup_modules": forbidden,
        "forbidden_startup_calls": forbidden_calls,
        "timings_seconds": timings,
    }
    if forbidden or forbidden_calls:
        raise RuntimeError(f"cold startup loaded execution paths: {report}")
    if median_seconds > _MEDIAN_LIMIT_SECONDS:
        raise RuntimeError(f"cold startup median exceeded budget: {report}")
    if p95_seconds > _P95_LIMIT_SECONDS:
        raise RuntimeError(f"cold startup p95 exceeded budget: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.state_dir), sort_keys=True))


if __name__ == "__main__":
    main()
