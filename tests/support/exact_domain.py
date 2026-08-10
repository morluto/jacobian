"""Shared harness for installing exact-domain checkers in domain tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from jacobian.exact_domain_checkers import install_exact_domain_verification
from jacobian.operations import DomainBundle
from jacobian.portfolio.domain_installation import DomainBundleInstaller
from jacobian.portfolio.model import PortfolioPlan
from jacobian.runtime.config import CheckerAuthorityMode
from tests.support.services import DomainTestServices, open_domain_services


@dataclass(frozen=True, slots=True)
class VerifiedDomainTestSpec:
    """Typed selection of one domain bundle with exact verification adapters.

    Bundle identity comes from authoritative ``DomainBundle.domain_id`` metadata;
    callers do not pass a parallel unvalidated string key.
    """

    bundle: DomainBundle

    @property
    def domain_id(self) -> str:
        return self.bundle.domain_id


def _as_specs(
    *items: VerifiedDomainTestSpec | DomainBundle,
) -> tuple[VerifiedDomainTestSpec, ...]:
    specs: list[VerifiedDomainTestSpec] = []
    for item in items:
        if isinstance(item, DomainBundle):
            specs.append(VerifiedDomainTestSpec(bundle=item))
        else:
            specs.append(item)
    return tuple(specs)


def install_verified_domain_bundles(
    services: DomainTestServices,
    *specs: VerifiedDomainTestSpec,
) -> None:
    """Install selected bundles and register their exact verification adapters.

    Uses the real ``DomainBundleInstaller`` and
    ``install_exact_domain_verification`` production paths. Checker authorization
    follows ``services.installation.authorizes_bundled_checkers``.
    """

    if not specs:
        raise ValueError("at least one verified domain bundle is required")
    installed = DomainBundleInstaller(services.installation).install(
        PortfolioPlan(domain_bundles=tuple(spec.bundle for spec in specs))
    )
    missing = tuple(
        spec.domain_id for spec in specs if spec.domain_id not in installed.installed
    )
    if missing:
        raise ValueError(
            "verified domain installation omitted bundle(s): " + ", ".join(missing)
        )
    bundles = {
        spec.domain_id: (spec.bundle, installed.installed[spec.domain_id])
        for spec in specs
    }
    adapters, exact = install_exact_domain_verification(
        services.core.store,
        services.core.schemas,
        services.core.artifacts,
        services.application.verification,
        services.core.checkers,
        bundles=bundles,
        authorize=services.installation.authorizes_bundled_checkers,
    )
    for adapter in adapters:
        services.installation.register_capability(adapter)
    for relationship in exact.catalog_relationships:
        services.installation.register_checker_relationship(
            relationship.source_capability_id,
            relationship.related_capability,
        )


@contextmanager
def open_exact_domain_services(
    root: str | Path,
    *bundles_or_specs: VerifiedDomainTestSpec | DomainBundle,
    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.INSTALL_BUNDLED,
) -> Iterator[DomainTestServices]:
    """Open domain services with explicitly selected verified domain bundles.

    Ordinary domain fixtures declare a bundle (or ``VerifiedDomainTestSpec``)
    rather than assembling the installer/checker/adapter-registration recipe.
    """

    specs = _as_specs(*bundles_or_specs)
    with open_domain_services(root, checker_authority=checker_authority) as services:
        install_verified_domain_bundles(services, *specs)
        yield services
