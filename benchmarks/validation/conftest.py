"""Keep benchmark validation imports rooted at the repository checkout.

Global autouse import scrubbing is intentionally avoided (#1170). Modules that
load task-local ``verifier_support`` opt into isolation via
``usefixtures("isolate_verifier_imports")`` or the collection rule below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ISOLATE_PREFIXES = (
    "benchmarks/validation/mathematical_benchmarks_v1/",
    "benchmarks/validation/symbolic_coordination_v1/",
)


@pytest.fixture
def isolate_verifier_imports():
    """Prevent one task's top-level verifier_support import leaking to another."""

    original_path = list(sys.path)
    sys.modules.pop("verifier_support", None)
    try:
        yield
    finally:
        sys.modules.pop("verifier_support", None)
        sys.path[:] = original_path


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    del config
    marker = pytest.mark.usefixtures("isolate_verifier_imports")
    for item in items:
        relative = Path(str(item.path)).as_posix()
        if (
            any(prefix in relative for prefix in _ISOLATE_PREFIXES)
            or "verifier_child" in relative
            or "test_verifier_" in Path(relative).name
        ):
            item.add_marker(marker)
