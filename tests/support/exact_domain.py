"""Shared harness for installing exact-domain checkers in domain tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from jacobian.operation_installation import InstalledDomainBundle
from jacobian.operations import DomainBundle
from jacobian.portfolio import (
    ApplicationInstallPlan,
    InstallationReceipt,
    open_application,
)
from jacobian.runtime.config import CheckerAuthorityMode
from tests.support.services import DomainTestServices


@dataclass(frozen=True, slots=True)
class VerifiedDomainTestServices(DomainTestServices):
    """Focused services plus the exact installed bundle resources."""

    bundles: dict[str, InstalledDomainBundle]
    plan: ApplicationInstallPlan
    receipt: InstallationReceipt


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
    with open_application(root, plan) as application:
        yield VerifiedDomainTestServices(
            core=application.core,
            application=application.services,
            installation=application.installation,
            bundles=dict(application.bundles),
            plan=plan,
            receipt=application.receipt,
        )
