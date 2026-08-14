"""Keep benchmark validation imports isolated at their exact dynamic boundaries."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

_ISOLATE_PREFIXES = (
    "benchmarks/validation/mathematical_benchmarks_v1/",
    "benchmarks/validation/symbolic_coordination_v1/",
    "benchmarks/validation/conjecture_probes_v1/",
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
