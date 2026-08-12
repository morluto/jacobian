"""Resolve packaged and operator-installed provider runtimes once."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from jacobian.contracts.capabilities import (
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
)
from jacobian.provider_runtime import (
    ProviderRuntimeError,
    ProviderRuntimeErrorCode,
    known_provider_runtime,
)
from jacobian.providers.external_solver_runtime import (
    cadical_provider_runtime,
    carcara_provider_runtime,
    cvc5_provider_runtime,
    drat_trim_provider_runtime,
)
from jacobian.providers.flint_runtime import (
    python_flint_provider_runtime,
)
from jacobian.providers.lean_runtime import (
    lean_frontend_provider_runtime,
    lean_provider_runtime,
)
from jacobian.providers.sympy_runtime import (
    sympy_polynomial_normalization_provider_runtime,
)


@dataclass(frozen=True, slots=True)
class ProviderRuntimePlan:
    """Resolved runtimes consumed by installation without repeating probes."""

    cadical: CapabilityProviderRuntime
    carcara: CapabilityProviderRuntime
    cvc5: CapabilityProviderRuntime
    drat_trim: CapabilityProviderRuntime
    sympy_polynomial_normalization: CapabilityProviderRuntime


@dataclass(frozen=True, slots=True)
class ProviderAvailabilityResolver:
    """Resolve provider availability before capability installation begins."""

    def resolve(self) -> ProviderRuntimePlan:
        cvc5 = cvc5_provider_runtime()
        sympy_polynomial_normalization = (
            sympy_polynomial_normalization_provider_runtime()
        )
        _require_packaged_python_backends(
            (
                known_provider_runtime("jacobian.networkx"),
                known_provider_runtime("jacobian.sympy"),
                known_provider_runtime("jacobian.z3"),
                python_flint_provider_runtime(),
                cvc5,
                sympy_polynomial_normalization,
            )
        )
        return ProviderRuntimePlan(
            cadical=cadical_provider_runtime(),
            carcara=carcara_provider_runtime(),
            cvc5=cvc5,
            drat_trim=drat_trim_provider_runtime(),
            sympy_polynomial_normalization=sympy_polynomial_normalization,
        )

    def resolve_lean(
        self,
        *,
        profiles: Mapping[str, Mapping[str, object]],
        checker_ids: tuple[str, ...],
    ) -> CapabilityProviderRuntime:
        """Resolve Lean after authorized checker profiles have been installed."""

        return lean_provider_runtime(profiles=profiles, checker_ids=checker_ids)

    def resolve_lean_frontend(self) -> CapabilityProviderRuntime:
        """Resolve the pinned CORE Lean frontend before statement registration."""

        return lean_frontend_provider_runtime()


def _require_packaged_python_backends(
    runtimes: tuple[CapabilityProviderRuntime, ...],
) -> None:
    """Reject an incomplete or version-skewed base installation."""

    failures: dict[str, str] = {}
    for runtime in runtimes:
        if runtime.availability is CapabilityProviderAvailability.AVAILABLE:
            continue
        detail = runtime.diagnostic or "the pinned runtime identity is unavailable"
        failures.setdefault(runtime.provider, detail[:256])
    if not failures:
        return
    diagnostic = "; ".join(
        f"{provider}: {detail}" for provider, detail in failures.items()
    )
    raise ProviderRuntimeError(
        f"required Python providers are unavailable: {diagnostic}",
        code=ProviderRuntimeErrorCode.UNAVAILABLE,
    )
