"""Explicit bundles for normalized Jacobian Nullstellensatz evidence."""

from __future__ import annotations

from collections.abc import Mapping

from jacobian.contracts.capabilities import (
    CapabilityDiagnostic,
    CapabilityProviderRuntime,
)
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
from jacobian.operations import DomainBundle, DomainDiagnostics, DomainSemantics
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
        raise ValueError("the Nullstellensatz core bundle has no dependencies")
    return install_nullstellensatz_core(context, _core_runtime())


def _install_singular(
    context: InstallationContext,
    dependencies: Mapping[str, InstalledDomainBundle],
) -> InstalledDomainBundle:
    if set(dependencies) != {CORE_DOMAIN_ID}:
        raise ValueError(
            "the Singular producer requires the Nullstellensatz core bundle"
        )
    runtime = singular_provider_runtime()
    return install_singular_producer(context, dependencies[CORE_DOMAIN_ID], runtime)


def _diagnostics() -> DomainDiagnostics:
    return DomainDiagnostics(
        invalid_request=CapabilityDiagnostic(
            code="INVALID_NULLSTELLENSATZ_REQUEST",
            stage="request_validation",
            message="The bounded Nullstellensatz request is invalid.",
            hint="Use producer-owned typed artifacts and the declared bounded schema.",
        )
    )


def build_nullstellensatz_core_bundle() -> DomainBundle:
    """Build the always-available system materializer and exact checker lane."""

    return DomainBundle(
        domain_id=CORE_DOMAIN_ID,
        schema_namespace="jacobian.nullstellensatz",
        semantics=DomainSemantics(
            name="jacobian.normalized-bivariate-jacobian-degree-2-3",
            version="1",
            definition={"managed_by": CORE_DOMAIN_ID},
        ),
        provider_runtime=_core_runtime(),
        backend_version="jacobian",
        capabilities=(),
        diagnostics=_diagnostics(),
        managed_capability_ids=(MATERIALIZE_CAPABILITY_ID, VERIFY_CAPABILITY_ID),
        managed_installer=_install_core,
    )


def build_nullstellensatz_singular_bundle() -> DomainBundle:
    """Build the optional pinned Singular certificate producer."""

    return DomainBundle(
        domain_id=SINGULAR_DOMAIN_ID,
        schema_namespace="jacobian.nullstellensatz.singular",
        semantics=DomainSemantics(
            name="jacobian.singular-nullstellensatz-producer",
            version="1",
            definition={"managed_by": SINGULAR_DOMAIN_ID},
        ),
        provider_runtime=singular_provider_runtime(),
        backend_version="4.4.1p5",
        capabilities=(),
        diagnostics=_diagnostics(),
        managed_capability_ids=(PRODUCE_CAPABILITY_ID,),
        managed_installer=_install_singular,
        dependency_ids=(CORE_DOMAIN_ID,),
    )


__all__ = [
    "CORE_DOMAIN_ID",
    "SINGULAR_DOMAIN_ID",
    "build_nullstellensatz_core_bundle",
    "build_nullstellensatz_singular_bundle",
]
