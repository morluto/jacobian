"""Readiness probes for the pinned external SAT toolchain."""


def external_sat_toolchain_available() -> bool:
    """Return whether the pinned CaDiCaL and DRAT-trim pair is usable.

    A binary on ``PATH`` is not sufficient for these tests: the production
    probes also validate version, provenance, and the checker health command.
    """

    from jacobian.contracts.operations import ProviderAvailability
    from jacobian.providers.external_solver_runtime import (
        cadical_provider_runtime,
        drat_trim_provider_runtime,
    )

    return all(
        runtime.availability is ProviderAvailability.AVAILABLE
        for runtime in (
            cadical_provider_runtime(),
            drat_trim_provider_runtime(),
        )
    )


def cadical_runtime_available() -> bool:
    """Return whether the pinned CaDiCaL executable passes readiness."""

    from jacobian.contracts.operations import ProviderAvailability
    from jacobian.providers.external_solver_runtime import cadical_provider_runtime

    return cadical_provider_runtime().availability is ProviderAvailability.AVAILABLE


def drat_trim_runtime_available() -> bool:
    """Return whether the pinned DRAT-trim checker passes readiness."""

    from jacobian.contracts.operations import ProviderAvailability
    from jacobian.providers.external_solver_runtime import drat_trim_provider_runtime

    return drat_trim_provider_runtime().availability is ProviderAvailability.AVAILABLE
