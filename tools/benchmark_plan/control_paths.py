"""Authoritative Harbor/benchmark control-plane path set.

Kept outside ``.github/scripts/plan-benchmarks`` so validation tests and the
planner share one constant without loading the CI adapter.
"""

from __future__ import annotations

BENCHMARK_CONTROL_PATHS = frozenset(
    {
        ".github/scripts/_ci_paths.py",
        ".github/scripts/plan-benchmarks",
        ".github/scripts/manage-test-timings",
        ".github/workflows/benchmarks.yml",
        ".github/workflows/heldout-benchmarks.yml",
        "Makefile",
        "make/harbor.mk",
        "benchmarks/tooling/validation_plan.py",
        "tools/benchmark_plan/__init__.py",
        "tools/benchmark_plan/compiler.py",
        "tools/benchmark_plan/control_paths.py",
        "tools/benchmark_plan/model.py",
        "tools/benchmark_plan/paths.py",
        "tools/benchmark_plan/validation.py",
        "tools/benchmark_pr_status.py",
        "tools/check_benchmark_adapters.py",
        "tools/check_benchmark_contracts.py",
        "tools/check_benchmark_static.py",
        "tools/check_harbor_dataset.py",
        "tools/harbor_task_workflow.py",
        "tools/process_supervisor.py",
        "tools/pytest_lifecycle.py",
        "tools/sync_harbor_verifier_support.py",
    }
)

__all__ = ["BENCHMARK_CONTROL_PATHS"]
