"""Harbor/benchmark plan compiler package.

Product test planning lives in ``tools.test_plan``. Benchmark planning mirrors
that ownership: typed helpers here, thin ``.github/scripts/plan-benchmarks``
adapters at the CI boundary.
"""

from __future__ import annotations

from tools.benchmark_plan.compiler import (
    PLAN_VERSION,
    PLANNER_DIGEST_SOURCES,
    plan,
)
from tools.benchmark_plan.control_paths import BENCHMARK_CONTROL_PATHS

__all__ = [
    "BENCHMARK_CONTROL_PATHS",
    "PLANNER_DIGEST_SOURCES",
    "PLAN_VERSION",
    "plan",
]
