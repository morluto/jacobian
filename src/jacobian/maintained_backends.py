"""Startup validation for mathematical libraries shipped with Jacobian."""

from __future__ import annotations

from jacobian.contracts.operations import ProviderAvailability, ProviderObservation
from jacobian.provider_runtime import (
    ProviderRuntimeError,
    ProviderRuntimeErrorCode,
    known_provider_runtime,
)
from jacobian.providers.external_solver_runtime import cvc5_provider_runtime
from jacobian.providers.flint_runtime import python_flint_provider_runtime


def require_maintained_math_backends() -> None:
    """Fail catalog compilation when a pinned Python math library is broken."""

    observations = (
        known_provider_runtime("jacobian.networkx"),
        known_provider_runtime("jacobian.sympy"),
        known_provider_runtime("jacobian.z3"),
        python_flint_provider_runtime(),
        cvc5_provider_runtime(),
    )
    failures = _unavailable_backends(observations)
    if not failures:
        return
    diagnostic = "; ".join(
        f"{backend}: {detail}" for backend, detail in failures.items()
    )
    raise ProviderRuntimeError(
        f"required Python math backends are unavailable: {diagnostic}",
        code=ProviderRuntimeErrorCode.UNAVAILABLE,
    )


def _unavailable_backends(
    observations: tuple[ProviderObservation, ...],
) -> dict[str, str]:
    failures: dict[str, str] = {}
    for observation in observations:
        if observation.availability is ProviderAvailability.AVAILABLE:
            continue
        detail = observation.diagnostic or "the pinned identity is unavailable"
        failures.setdefault(observation.provider, detail[:256])
    return failures


__all__ = ["require_maintained_math_backends"]
