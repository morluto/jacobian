"""Readiness probe for the pinned Lean and Mathlib test runtime."""

from __future__ import annotations

import os

from jacobian.contracts.capabilities import CapabilityProviderAvailability

PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON = (
    "the pinned Lean and Mathlib runtime is unavailable"
)
PINNED_LEAN_CORE_RUNTIME_UNAVAILABLE_REASON = (
    "the pinned Lean CORE runtime is unavailable"
)
_HOSTED_LEAN_REQUIRED = "JACOBIAN_LEAN_REQUIRED"


def hosted_lean_runtime_required() -> bool:
    """Return whether missing Lean runtimes must fail instead of skip."""

    return os.environ.get(_HOSTED_LEAN_REQUIRED) == "1"


def pinned_lean_core_runtime_diagnostic() -> str | None:
    """Return the CORE frontend diagnostic when that runtime is unavailable."""

    from jacobian.providers.lean_runtime import lean_frontend_provider_runtime

    runtime = lean_frontend_provider_runtime()
    if runtime.availability is CapabilityProviderAvailability.AVAILABLE:
        return None
    return runtime.diagnostic or PINNED_LEAN_CORE_RUNTIME_UNAVAILABLE_REASON


def pinned_mathlib_runtime_diagnostic() -> str | None:
    """Return the Mathlib runtime diagnostic when that runtime is unavailable."""

    from jacobian.providers.lean_runtime import lean_provider_runtime

    runtime = lean_provider_runtime(
        profiles={"mathlib": {"mathlib_commit": "pinned"}},
        checker_ids=(),
    )
    if runtime.availability is CapabilityProviderAvailability.AVAILABLE:
        return None
    return runtime.diagnostic or PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON


def _skip_unavailable(diagnostic: str | None, fallback: str) -> bool:
    if diagnostic is None:
        return False
    if hosted_lean_runtime_required():
        raise RuntimeError(diagnostic)
    del fallback
    return True


def pinned_lean_core_runtime_available() -> bool:
    """Return whether the production pinned Lean CORE frontend probe succeeds."""

    return pinned_lean_core_runtime_diagnostic() is None


def pinned_mathlib_runtime_available() -> bool:
    """Return whether the production pinned Lean/Mathlib probe succeeds.

    The implementation is imported lazily so this support module remains safe
    during collection of unit and component tests.
    """

    return pinned_mathlib_runtime_diagnostic() is None


def skip_unless_pinned_lean_core_runtime() -> bool:
    """Skip CORE Lean tests locally; fail when the hosted lane requires them."""

    return _skip_unavailable(
        pinned_lean_core_runtime_diagnostic(),
        PINNED_LEAN_CORE_RUNTIME_UNAVAILABLE_REASON,
    )


def skip_unless_pinned_mathlib_runtime() -> bool:
    """Skip Mathlib Lean tests locally; fail when the hosted lane requires them."""

    return _skip_unavailable(
        pinned_mathlib_runtime_diagnostic(),
        PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON,
    )
