#!/usr/bin/env python3
"""Write one bounded timing receipt for an ordinary CI test lane."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from pathlib import Path
from typing import Any


def read_metrics(path: Path) -> dict[str, float | int]:
    """Parse the fixed GNU time output emitted by the CI action."""

    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields or not value:
            raise ValueError(f"invalid timing metric in {path}: {line!r}")
        fields[key] = value
    if set(fields) != {"wall_seconds", "peak_rss_kib"}:
        raise ValueError(f"unexpected timing metrics in {path}")
    wall_seconds = float(fields["wall_seconds"])
    peak_rss_kib = int(fields["peak_rss_kib"])
    if not math.isfinite(wall_seconds) or wall_seconds < 0 or peak_rss_kib < 0:
        raise ValueError(f"invalid timing values in {path}")
    return {"wall_seconds": wall_seconds, "peak_rss_kib": peak_rss_kib}


def timing_receipt(
    *,
    lane: str,
    worker_count: int,
    collection: dict[str, float | int],
    execution: dict[str, float | int],
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    if worker_count < 1:
        raise ValueError("worker count must be positive")
    measured_environment = environment or os.environ
    return {
        "version": 1,
        "lane": lane,
        "worker_count": worker_count,
        "collection": collection,
        "execution": execution,
        "environment": {
            "event": measured_environment.get("GITHUB_EVENT_NAME", "local"),
            "revision": measured_environment.get("GITHUB_SHA", "unknown"),
            "runner_os": measured_environment.get("RUNNER_OS", platform.system()),
            "runner_arch": measured_environment.get("RUNNER_ARCH", platform.machine()),
            "python_version": platform.python_version(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", default=os.environ.get("LANE"), required=False)
    parser.add_argument(
        "--worker-count",
        type=int,
        default=os.environ.get("WORKER_COUNT"),
        required=False,
    )
    arguments = parser.parse_args()
    if not arguments.lane:
        parser.error("--lane or LANE is required")
    if arguments.worker_count is None:
        parser.error("--worker-count or WORKER_COUNT is required")
    receipt = timing_receipt(
        lane=arguments.lane,
        worker_count=arguments.worker_count,
        collection=read_metrics(arguments.collection),
        execution=read_metrics(arguments.execution),
    )
    arguments.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
