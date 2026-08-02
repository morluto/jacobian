"""Collect per-task Harbor Oracle wall times for future deterministic sharding."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.tooling.harbor_suite import HarborSuiteError, load_registry


def _seconds(started: Any, finished: Any) -> float | None:
    if not isinstance(started, str) or not isinstance(finished, str):
        return None
    try:
        start = datetime.fromisoformat(started.replace("Z", "+00:00"))
        finish = datetime.fromisoformat(finished.replace("Z", "+00:00"))
    except ValueError:
        return None
    value = (finish - start).total_seconds()
    return value if value > 0 else None


def collect(root: Path) -> dict[str, float]:
    owners = {
        ref.path.name: suite.id for suite in load_registry() for ref in suite.tasks
    }
    samples: dict[str, list[float]] = {}
    for path in sorted(root.rglob("result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or "task_name" not in payload:
            continue
        task = str(payload["task_name"]).rsplit("/", 1)[-1]
        dataset = owners.get(task)
        elapsed = _seconds(payload.get("started_at"), payload.get("finished_at"))
        if dataset is None or elapsed is None:
            continue
        samples.setdefault(f"{dataset}/{task}", []).append(elapsed)
    if not samples:
        raise HarborSuiteError(f"no completed Harbor trial timings found below {root}")
    return {
        key: round(statistics.median(values), 6)
        for key, values in sorted(samples.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        timings = collect(args.root)
    except HarborSuiteError as exc:
        parser.error(str(exc))
    args.output.write_text(
        json.dumps(timings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["collect"]
