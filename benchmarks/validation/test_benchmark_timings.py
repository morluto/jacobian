from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.tooling.benchmark_timings import collect
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.harbor_suite import get_suite


def test_timing_collector_uses_median_completed_trial_duration(tmp_path: Path) -> None:
    task = get_suite("mathematical-benchmarks-v1").tasks[0].path.name
    for index, minute in enumerate((1, 3, 2)):
        path = tmp_path / str(index) / "result.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "task_name": f"jacobian/{task}",
                    "started_at": "2026-01-01T00:00:00Z",
                    "finished_at": f"2026-01-01T00:0{minute}:00Z",
                }
            ),
            encoding="utf-8",
        )

    timings = collect(tmp_path)

    assert timings[f"mathematical-benchmarks-v1/{task}"] == 120.0


def test_timing_collector_fails_closed_without_trials(tmp_path: Path) -> None:
    with pytest.raises(HarborSuiteError, match="no completed"):
        collect(tmp_path)
