"""Shared harness for installing exact-domain checkers in domain tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from jacobian.exact_domain_checkers import install_exact_domain_verification
from jacobian.operation_installation import InstalledDomainBundle
from jacobian.operations import DomainBundle
from jacobian.portfolio.application_plan import (
    ApplicationInstallPlan,
    InstallationReceipt,
    receipt_from_installed_bundles,
)
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
    plan: ApplicationInstallPlan
    receipt: InstallationReceipt


def install_verified_domain_bundles(
    services: DomainTestServices,
    *bundles: DomainBundle,
) -> tuple[dict[str, InstalledDomainBundle], tuple[str, ...]]:
    """Install selected bundles and register their exact verification adapters.

    Uses the real ``DomainBundleInstaller`` and
    ``install_exact_domain_verification`` production paths. Checker authorization
    follows ``services.installation.authorizes_bundled_checkers``.

    Returns installed bundles and the checker IDs registered for the receipt.
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
        adapters, exact = install_exact_domain_verification(
            services.core.store,
            services.core.schemas,
            services.core.artifacts,
            services.application.verification,
            services.core.checkers,
            bundles=installed_bundles,
            authorize=services.installation.authorizes_bundled_checkers,
        )
        for adapter in adapters:
            services.installation.register_capability(adapter)
    checker_ids = tuple(
        checker_id
        for checker_id in exact.checker_ids.values()
        if isinstance(checker_id, str) and checker_id
    )
    return installed.installed, checker_ids


@contextmanager
def open_exact_domain_services(
    root: str | Path,
    *bundles: DomainBundle,
    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.INSTALL_BUNDLED,
) -> Iterator[VerifiedDomainTestServices]:
    """Open domain services with explicitly selected verified domain bundles.

    Ordinary domain fixtures declare their bundles rather than assembling the
    installer/checker/adapter-registration recipe. Every successful open emits
    an ``InstallationReceipt`` from the shared application plan vocabulary.
    """

    if not bundles:
        raise ValueError("at least one verified domain bundle is required")
    plan = ApplicationInstallPlan.scoped(
        tuple(bundle.domain_id for bundle in bundles),
        checker_authority=checker_authority,
        include_exact_verification=True,
    )
    with open_domain_services(root, checker_authority=checker_authority) as services:
        installed, checker_ids = install_verified_domain_bundles(services, *bundles)
        receipt = receipt_from_installed_bundles(
            plan,
            installed,
            checker_ids=checker_ids,
        )
        yield VerifiedDomainTestServices(
            core=services.core,
            application=services.application,
            installation=services.installation,
            bundles=installed,
            plan=plan,
            receipt=receipt,
        )
