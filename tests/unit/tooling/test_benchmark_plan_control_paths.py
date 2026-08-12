"""Unit tests for shared Harbor/benchmark control paths."""

from __future__ import annotations

from tools.benchmark_plan import BENCHMARK_CONTROL_PATHS


def test_benchmark_control_paths_include_planner_and_package() -> None:
    assert ".github/scripts/plan-benchmarks" in BENCHMARK_CONTROL_PATHS
    assert "tools/benchmark_plan/control_paths.py" in BENCHMARK_CONTROL_PATHS
    assert "tools/benchmark_plan/validation.py" in BENCHMARK_CONTROL_PATHS
    assert "tools/test_plan/affinity.py" in BENCHMARK_CONTROL_PATHS
    assert "Makefile" in BENCHMARK_CONTROL_PATHS
