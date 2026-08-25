from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[2]


def _load() -> ModuleType:
    path = ROOT / ".github" / "scripts" / "record_test_lane_timing.py"
    spec = importlib.util.spec_from_file_location("record_test_lane_timing", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_timing_receipt_separates_collection_execution_and_environment() -> None:
    timing = _load()

    receipt = timing.timing_receipt(
        lane="math",
        worker_count=4,
        collection={"wall_seconds": 3.5, "peak_rss_kib": 120_000},
        execution={"wall_seconds": 12.75, "peak_rss_kib": 240_000},
        environment={
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_SHA": "abc123",
            "RUNNER_OS": "Linux",
            "RUNNER_ARCH": "X64",
        },
    )

    assert receipt["lane"] == "math"
    assert receipt["worker_count"] == 4
    assert receipt["collection"]["wall_seconds"] == 3.5
    assert receipt["execution"]["peak_rss_kib"] == 240_000
    assert receipt["environment"]["event"] == "pull_request"
    assert receipt["environment"]["revision"] == "abc123"


def test_read_metrics_rejects_missing_or_invalid_fields(tmp_path: Path) -> None:
    timing = _load()
    metrics = tmp_path / "metrics"
    metrics.write_text("wall_seconds=1.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected timing metrics"):
        timing.read_metrics(metrics)

    metrics.write_text("wall_seconds=nan\npeak_rss_kib=1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid timing values"):
        timing.read_metrics(metrics)
