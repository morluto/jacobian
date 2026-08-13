"""Keep benchmark validation imports isolated at their exact dynamic boundaries."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
import tools.benchmark_plan.compiler as benchmark_planner
from tools.benchmark_plan.validation import validate_plan

_ISOLATE_PREFIXES = (
    "benchmarks/validation/mathematical_benchmarks_v1/",
    "benchmarks/validation/symbolic_coordination_v1/",
    "benchmarks/validation/conjecture_probes_v1/",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Bind the legacy planner corpus to the extracted package owner.

    The corpus remains at the explicit CLI-integration path allowed to exercise
    SourceFileLoader and subprocess seams. Its semantic calls, monkeypatches,
    and plan validation are redirected to the importable package modules after
    collection. The obsolete three-file digest assertion is replaced by the
    extraction-specific regression module.
    """

    rebound_modules: set[ModuleType] = set()
    for item in items:
        if Path(str(item.path)).name != "test_benchmark_planner.py":
            continue
        module = getattr(item, "module", None)
        if isinstance(module, ModuleType) and module not in rebound_modules:
            module.__dict__["planner"] = benchmark_planner
            module.__dict__["_assert_plan_valid"] = validate_plan
            rebound_modules.add(module)
        if item.name == "test_planner_digest_binds_to_planner_and_path_policy_sources":
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "replaced by the package-source digest regression in "
                        "test_benchmark_planner_extraction.py"
                    )
                )
            )


@pytest.fixture(autouse=True)
def isolate_verifier_imports(request: pytest.FixtureRequest) -> Iterator[None]:
    """Prevent one task's top-level verifier_support import leaking to another."""

    relative = Path(str(request.node.path)).as_posix()
    needs_isolation = (
        any(prefix in relative for prefix in _ISOLATE_PREFIXES)
        or "verifier_child" in relative
        or "test_verifier_" in Path(relative).name
    )
    if not needs_isolation:
        yield
        return

    original_path = list(sys.path)
    sys.modules.pop("verifier_support", None)
    try:
        yield
    finally:
        sys.modules.pop("verifier_support", None)
        sys.path[:] = original_path
