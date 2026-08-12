"""Minimal installers for core finite/universal-algebra capability tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from jacobian.finite_coverage import (
    FiniteCoverageInstallation,
    install_finite_coverage,
)
from jacobian.graphs import GraphInstallation, install_graph_capabilities
from jacobian.runtime.config import CheckerAuthorityMode
from jacobian.sat_smt.sat_capabilities import SatCnfMaterializationAdapter
from jacobian.universal_algebra_capabilities import (
    UniversalAlgebraInstallation,
    install_universal_algebra_capabilities,
)
from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)


@dataclass(frozen=True, slots=True)
class FiniteCoverageTestServices:
    services: DomainTestServices
    installation: FiniteCoverageInstallation


@dataclass(frozen=True, slots=True)
class UniversalAlgebraTestServices:
    services: DomainTestServices
    installation: UniversalAlgebraInstallation


def _authority(authorize_checker: bool) -> CheckerAuthorityMode:
    return (
        CheckerAuthorityMode.INSTALL_BUNDLED
        if authorize_checker
        else CheckerAuthorityMode.NONE
    )


@contextmanager
def open_finite_coverage_services(
    root: str | Path,
    *,
    authorize_checker: bool = True,
) -> Iterator[FiniteCoverageTestServices]:
    """Install only finite coverage verification into a domain service graph."""

    with open_domain_services(
        root,
        checker_authority=_authority(authorize_checker),
    ) as services:
        with atomic_installation(services.core):
            adapter, installation = install_finite_coverage(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.application.verification,
                services.core.checkers,
                authorize_checker=services.installation.authorizes_bundled_checkers,
            )
            if adapter is not None:
                services.installation.register_capability(adapter)
        yield FiniteCoverageTestServices(services=services, installation=installation)


@contextmanager
def open_universal_algebra_services(
    root: str | Path,
    *,
    authorize_checker: bool = True,
) -> Iterator[UniversalAlgebraTestServices]:
    """Install only universal-algebra capabilities into a domain service graph."""

    with open_domain_services(
        root,
        checker_authority=_authority(authorize_checker),
    ) as services:
        with atomic_installation(services.core):
            adapters, installation = install_universal_algebra_capabilities(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.application.verification,
                services.core.checkers,
                authorize_checker=services.installation.authorizes_bundled_checkers,
            )
            for adapter in adapters:
                services.installation.register_capability(adapter)
        yield UniversalAlgebraTestServices(services=services, installation=installation)


@contextmanager
def open_graph_core_services(
    root: str | Path,
) -> Iterator[tuple[DomainTestServices, GraphInstallation]]:
    """Install core graph construction/search/property capabilities only."""

    with open_domain_services(root) as services:
        with atomic_installation(services.core):
            adapters, installation = install_graph_capabilities(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.application.verification,
                services.core.checkers,
                authorize_checker=False,
            )
            for adapter in adapters:
                services.installation.register_capability(adapter)
        yield services, installation


@contextmanager
def open_sat_materialization_services(
    root: str | Path,
) -> Iterator[DomainTestServices]:
    """Register sat.cnf.materialize on a domain service graph."""

    with open_domain_services(root) as services:
        with atomic_installation(services.core):
            services.installation.register_capability(
                SatCnfMaterializationAdapter(services.core.sat)
            )
        yield services
