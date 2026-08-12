"""Portfolio installation for Nullstellensatz artifact and checker operations."""

from __future__ import annotations

from collections.abc import Mapping

from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.domains.polynomial_nullstellensatz.core import (
    MATERIALIZE_CAPABILITY_ID,
    VERIFY_CAPABILITY_ID,
    install_nullstellensatz_core,
)
from jacobian.domains.polynomial_nullstellensatz.singular import (
    PRODUCE_CAPABILITY_ID,
    install_singular_producer,
)
from jacobian.installation.context import InstallationContext
from jacobian.operation_installation import InstalledDomainBundle
from jacobian.portfolio.model import ManagedPortfolioComponent
from jacobian.provider_runtime import known_provider_runtime
from jacobian.providers.singular_runtime import singular_provider_runtime

CORE_DOMAIN_ID = "polynomial_nullstellensatz"
SINGULAR_DOMAIN_ID = "polynomial_nullstellensatz_singular"


def _core_runtime() -> CapabilityProviderRuntime:
    return known_provider_runtime(
        "jacobian.nullstellensatz-core",
        features=(
            "normalized-jacobian-degree-slice",
            "rabinowitsch-chart-cover",
            "independent-exact-replay",
        ),
    )


def _install_core(
    context: InstallationContext,
    dependencies: Mapping[str, InstalledDomainBundle],
) -> InstalledDomainBundle:
    if dependencies:
        raise ValueError("the Nullstellensatz core has no dependencies")
    return install_nullstellensatz_core(context, _core_runtime())


def _install_singular(
    context: InstallationContext,
    dependencies: Mapping[str, InstalledDomainBundle],
) -> InstalledDomainBundle:
    if set(dependencies) != {CORE_DOMAIN_ID}:
        raise ValueError("the Singular producer requires the Nullstellensatz core")
    runtime = singular_provider_runtime()
    return install_singular_producer(context, dependencies[CORE_DOMAIN_ID], runtime)


def build_nullstellensatz_core_component() -> ManagedPortfolioComponent:
    """Build the always-available materializer and exact checker component."""

    return ManagedPortfolioComponent(
        domain_id=CORE_DOMAIN_ID,
        provider_runtime=_core_runtime(),
        capability_ids=(MATERIALIZE_CAPABILITY_ID, VERIFY_CAPABILITY_ID),
        install=_install_core,
    )


def build_nullstellensatz_singular_component() -> ManagedPortfolioComponent:
    """Build the pinned Singular certificate-producer component."""

    return ManagedPortfolioComponent(
        domain_id=SINGULAR_DOMAIN_ID,
        provider_runtime=singular_provider_runtime(),
        capability_ids=(PRODUCE_CAPABILITY_ID,),
        install=_install_singular,
        dependency_ids=(CORE_DOMAIN_ID,),
    )


__all__ = [
    "CORE_DOMAIN_ID",
    "SINGULAR_DOMAIN_ID",
    "build_nullstellensatz_core_component",
    "build_nullstellensatz_singular_component",
]
