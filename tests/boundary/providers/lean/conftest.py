"""Resource-aware readiness fixtures for the pinned Lean boundary."""

from __future__ import annotations

import pytest
from tests.support.provider_lean import (
    PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON,
    pinned_mathlib_runtime_available,
)
from tests.support.resource_contracts import (
    IsolationClass,
    ResourceKind,
    resource_fixture,
)


@pytest.fixture
@resource_fixture(
    resources={ResourceKind.LEAN, ResourceKind.PROVIDER},
    isolation=IsolationClass.LIFECYCLE_OWNER,
    setup_affinity="lean",
)
def lean_mathlib_ready() -> None:
    """Skip consumers unless the pinned Lean and Mathlib runtime is available."""

    if not pinned_mathlib_runtime_available():
        pytest.skip(PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON)


__all__ = ["lean_mathlib_ready"]
