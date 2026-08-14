"""Minimal installers for core finite/universal-algebra operation tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from jacobian.finite_coverage import (
    FiniteCoverageInstallation,
    install_finite_coverage,
)
from jacobian.graphs import GraphOperationResources, build_graph_operations
from jacobian.sat_smt.sat_operations import SatCnfMaterializationAdapter
from jacobian.universal_algebra_operations import (
    UniversalAlgebraInstallation,
    install_universal_algebra_operations,
)
from tests.support.catalog_build_options import CheckerAuthorityMode
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
                services.verification,
                services.core.checkers,
                authorize_checker=services.installation.authorize_bundled_checkers,
            )
            if adapter is not None:
                services.installation.register_operation(adapter)
        yield FiniteCoverageTestServices(services=services, installation=installation)


@contextmanager
def open_universal_algebra_services(
    root: str | Path,
    *,
    authorize_checker: bool = True,
) -> Iterator[UniversalAlgebraTestServices]:
    """Install only universal-algebra operations into a domain service graph."""

    with open_domain_services(
        root,
        checker_authority=_authority(authorize_checker),
    ) as services:
        with atomic_installation(services.core):
            adapters, installation = install_universal_algebra_operations(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.verification,
                services.core.checkers,
                authorize_checker=services.installation.authorize_bundled_checkers,
            )
            for adapter in adapters:
                services.installation.register_operation(adapter)
        yield UniversalAlgebraTestServices(services=services, installation=installation)


@contextmanager
def open_graph_core_services(
    root: str | Path,
) -> Iterator[tuple[DomainTestServices, GraphOperationResources]]:
    """Build core graph construction/search/property operations only."""

    with open_domain_services(root) as services:
        with atomic_installation(services.core):
            adapters, graph_resources = build_graph_operations(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.verification,
                services.core.checkers,
                authorize_checker=False,
            )
            for adapter in adapters:
                services.installation.register_operation(adapter)
        yield services, graph_resources


@contextmanager
def open_sat_materialization_services(
    root: str | Path,
) -> Iterator[DomainTestServices]:
    """Register sat.cnf.materialize on a domain service graph."""

    with open_domain_services(root) as services:
        with atomic_installation(services.core):
            services.installation.register_operation(
                SatCnfMaterializationAdapter(services.core.sat)
            )
        yield services
