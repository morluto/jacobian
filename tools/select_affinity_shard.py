#!/usr/bin/env python3
"""Collect a pytest suite and select one affinity-aware CI shard."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.test_plan.affinity import assign_collected_to_shards  # noqa: E402

DEFAULT_TOPOLOGY = ROOT / "tests" / "topology.toml"
SUITES = ("domain", "composition")


def suite_paths(suite: str, topology_path: Path = DEFAULT_TOPOLOGY) -> tuple[str, ...]:
    """Return the configured paths for one timing-sharded suite."""

    payload = tomllib.loads(topology_path.read_text(encoding="utf-8"))
    lanes = payload.get("lanes")
    if not isinstance(lanes, list):
        raise ValueError(f"{topology_path} does not define topology lanes")
    for lane in lanes:
        if not isinstance(lane, dict) or lane.get("name") != suite:
            continue
        paths = lane.get("paths")
        if (
            not isinstance(paths, list)
            or not paths
            or not all(isinstance(path, str) and path for path in paths)
        ):
            raise ValueError(f"{topology_path} has invalid paths for suite {suite}")
        return tuple(paths)
    raise ValueError(f"{topology_path} does not define suite {suite}")


def parse_collected_nodeids(output: str) -> tuple[str, ...]:
    """Extract unique pytest node IDs from quiet collection output."""

    return tuple(
        dict.fromkeys(line.strip() for line in output.splitlines() if "::" in line)
    )


def collect_nodeids(paths: tuple[str, ...]) -> tuple[str, ...]:
    """Collect node IDs from pytest without executing tests."""

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *paths],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, end="", file=sys.stderr)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            f"pytest collection failed with exit code {result.returncode}"
        )
    return parse_collected_nodeids(result.stdout)


def load_durations(path: Path | None) -> dict[str, float]:
    """Load a bare duration map or a timing-artifact duration envelope."""

    if path is None or not path.exists():
        return {}
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    raw_durations = payload.get("durations", payload)
    if not isinstance(raw_durations, dict):
        raise ValueError(f"{path} durations must be a JSON object")

    durations: dict[str, float] = {}
    for nodeid, duration in raw_durations.items():
        if not isinstance(nodeid, str):
            raise ValueError(f"{path} contains a non-string node ID")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError(f"{path} contains a non-numeric duration for {nodeid}")
        seconds = float(duration)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"{path} contains an invalid duration for {nodeid}")
        durations[nodeid] = seconds
    return durations


def write_nodeids(path: Path, nodeids: tuple[str, ...]) -> None:
    """Write one node ID per line, leaving empty shards as empty files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    contents = "".join(f"{nodeid}\n" for nodeid in nodeids)
    path.write_text(contents, encoding="utf-8")


def write_shard_plan(
    path: Path,
    *,
    suite: str,
    shards: tuple[tuple[str, ...], ...],
) -> None:
    """Write the complete affinity shard plan for diagnostics."""

    payload = {
        "version": 1,
        "suite": suite,
        "shard_count": len(shards),
        "shards": [list(shard) for shard in shards],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--shard", type=int, required=True, help="1-based shard index")
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument(
        "--paths",
        action="append",
        help="pytest path to collect; repeat to override the suite topology paths",
    )
    parser.add_argument("--durations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-shards", type=Path)
    args = parser.parse_args(argv)

    if args.shard_count <= 0:
        parser.error("--shard-count must be positive")
    if args.shard <= 0 or args.shard > args.shard_count:
        parser.error("--shard must be between 1 and --shard-count")

    try:
        paths = tuple(args.paths) if args.paths else suite_paths(args.suite)
        collected = collect_nodeids(paths)
        durations = load_durations(args.durations)
        shards = assign_collected_to_shards(
            collected,
            suite=args.suite,
            shard_count=args.shard_count,
            durations=durations,
        )
        selected = shards[args.shard - 1]
        write_nodeids(args.output, selected)
        if args.json_shards is not None:
            write_shard_plan(args.json_shards, suite=args.suite, shards=shards)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(
        f"suite={args.suite} shard={args.shard}/{args.shard_count} "
        f"collected={len(collected)} selected={len(selected)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
