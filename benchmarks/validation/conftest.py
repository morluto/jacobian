"""Keep benchmark validation imports rooted at the repository checkout."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate_task_verifier_imports():
    """Prevent one task's top-level verifier_support import leaking to another."""

    original_path = list(sys.path)
    sys.modules.pop("verifier_support", None)
    try:
        yield
    finally:
        sys.modules.pop("verifier_support", None)
        sys.path[:] = original_path
