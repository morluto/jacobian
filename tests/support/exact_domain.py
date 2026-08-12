"""Shared harness for installing exact-domain checkers in domain tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from jacobian.domain_bundles import DomainBundle
from jacobian.exact_domain_checkers import install_exact_domain_verification
from jacobian.operation_installation import InstalledDomainBundle
from jacobian.portfolio.domain_installation import DomainBundleInstaller
from jacobian.portfolio.model import PortfolioPlan
from jacobian.runtime.config import CheckerAuthorityMode
from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)


@dataclass(frozen=True, slots=True)
class VerifiedDomainTestServices(DomainTestServices):
    """Focused services plus the exact installed bundle resources."""

    bundles: dict[str, InstalledDomainBundle]


def install_verified_domain_bundles(
    services: DomainTestServices,
    *bundles: DomainBundle,
) -> dict[str, InstalledDomainBundle]:
    """Install selected bundles and register their exact verification adapters.

    Uses the real ``DomainBundleInstaller`` and
    ``install_exact_domain_verification`` production paths. Checker authorization
    follows ``services.installation.authorizes_bundled_checkers``.
    """

    if not bundles:
        raise ValueError("at least one verified domain bundle is required")
    with atomic_installation(services.core):
        installed = DomainBundleInstaller(services.installation).install(
            PortfolioPlan(components=bundles)
        )
        missing = tuple(
            bundle.domain_id
            for bundle in bundles
            if bundle.domain_id not in installed.installed
        )
        if missing:
            raise ValueError(
                "verified domain installation omitted bundle(s): " + ", ".join(missing)
            )
        installed_bundles = {
            bundle.domain_id: (bundle, installed.installed[bundle.domain_id])
            for bundle in bundles
        }
        adapters, _ = install_exact_domain_verification(
            services.core.store,
            services.core.schemas,
            services.core.artifacts,
            services.core.values,
            services.application.verification,
            services.core.checkers,
            bundles=installed_bundles,
            authorize=services.installation.authorizes_bundled_checkers,
        )
        for adapter in adapters:
            services.installation.register_capability(adapter)
    return installed.installed


@contextmanager
def open_exact_domain_services(
    root: str | Path,
    *bundles: DomainBundle,
    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.INSTALL_BUNDLED,
) -> Iterator[VerifiedDomainTestServices]:
    """Open domain services with explicitly selected verified domain bundles.

    Ordinary domain fixtures declare their bundles rather than assembling the
    installer/checker/adapter-registration recipe.
    """

    with open_domain_services(root, checker_authority=checker_authority) as services:
        installed = install_verified_domain_bundles(services, *bundles)
        yield VerifiedDomainTestServices(
            core=services.core,
            application=services.application,
            installation=services.installation,
            bundles=installed,
        )
