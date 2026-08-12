"""Authoritative Harbor/benchmark control-plane path set.

Kept outside ``.github/scripts/plan-benchmarks`` so validation tests and the
planner share one typed constant without loading the full CI script.
"""

from __future__ import annotations

BENCHMARK_CONTROL_PATHS = frozenset(
    {
        ".github/scripts/_ci_paths.py",
        ".github/scripts/emit-plan-receipt",
        ".github/scripts/plan-benchmarks",
        ".github/scripts/manage-test-timings",
        ".github/scripts/validate-benchmark-plan",
        ".github/workflows/benchmarks.yml",
        ".github/workflows/heldout-benchmarks.yml",
        "Makefile",
        "make/harbor.mk",
        "tools/check_benchmark_adapters.py",
        "tools/benchmark_pr_status.py",
        "tools/check_benchmark_contracts.py",
        "tools/check_benchmark_static.py",
        "tools/check_harbor_dataset.py",
        "tools/harbor_task_workflow.py",
        "tools/pytest_lifecycle.py",
        "benchmarks/tooling/validation_plan.py",
        "tools/sync_harbor_verifier_support.py",
        "tools/benchmark_plan/__init__.py",
        "tools/benchmark_plan/control_paths.py",
    }
)

__all__ = ["BENCHMARK_CONTROL_PATHS"]
