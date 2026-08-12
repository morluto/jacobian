"""Keep benchmark validation imports rooted at the repository checkout.

Task-local ``verifier_support`` modules are isolated only for validation tests
that dynamically load benchmark verifiers. Other tests retain normal import
caching and do not pay for global module scrubbing.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import tools.benchmark_plan.compiler as planner

ROOT = Path(__file__).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ISOLATE_PREFIXES = (
    "benchmarks/validation/mathematical_benchmarks_v1/",
    "benchmarks/validation/symbolic_coordination_v1/",
)

_PLANNER_TEST_MARKERS = (
    "test_benchmark_planner_classify.py",
    "test_benchmark_planner_digests.py",
    "test_benchmark_planner_host.py",
    "test_benchmark_planner_oracle.py",
)


@pytest.fixture(autouse=True)
def stable_planner_digests(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep planner tests independent of Harbor's optional runtime package."""

    path = Path(str(request.node.path)).name
    if path not in _PLANNER_TEST_MARKERS:
        return
    monkeypatch.setattr(
        planner,
        "_digest",
        lambda digest_path: (
            f"sha256:{hashlib.sha256(digest_path.name.encode()).hexdigest()}"
        ),
    )


@pytest.fixture(autouse=True)
def isolate_verifier_imports(request: pytest.FixtureRequest):
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
