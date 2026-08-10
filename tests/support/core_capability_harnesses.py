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
from jacobian.finite_partition import (
    FinitePartitionInstallation,
    install_finite_partition,
)
from jacobian.runtime.config import CheckerAuthorityMode
from jacobian.universal_algebra_capabilities import (
    UniversalAlgebraInstallation,
    install_universal_algebra_capabilities,
)
from tests.support.services import DomainTestServices, open_domain_services


@dataclass(frozen=True, slots=True)
class FiniteCoverageTestServices:
    services: DomainTestServices
    installation: FiniteCoverageInstallation


@dataclass(frozen=True, slots=True)
class FinitePartitionTestServices:
    services: DomainTestServices
    installation: FinitePartitionInstallation


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
def open_finite_partition_services(
    root: str | Path,
    *,
    authorize_checker: bool = True,
) -> Iterator[FinitePartitionTestServices]:
    """Install only finite partition producer/verify into a domain service graph."""

    with open_domain_services(
        root,
        checker_authority=_authority(authorize_checker),
    ) as services:
        producer, verify, installation = install_finite_partition(
            services.core.store,
            services.core.schemas,
            services.core.artifacts,
            services.application.verification,
            services.core.checkers,
            authorize_checker=services.installation.authorizes_bundled_checkers,
        )
        services.installation.register_capability(producer)
        if verify is not None:
            services.installation.register_capability(verify)
        yield FinitePartitionTestServices(services=services, installation=installation)


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
        adapters, installation = install_universal_algebra_capabilities(
            services.core.store,
            services.core.schemas,
            services.core.artifacts,
            services.core.checkers,
            authorize_checker=services.installation.authorizes_bundled_checkers,
        )
        for adapter in adapters:
            services.installation.register_capability(adapter)
        yield UniversalAlgebraTestServices(services=services, installation=installation)
