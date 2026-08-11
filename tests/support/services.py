"""Tier-local service graph helpers backed by production composition seams."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jacobian.implementation import cached_package_digests
from jacobian.installation.context import (
    InstallationContext,
    create_installation_context,
)
from jacobian.operations import DomainBundle
from jacobian.portfolio.domain_installation import DomainBundleInstaller
from jacobian.portfolio.model import PortfolioPlan
from jacobian.references import ReferenceInstallation
from jacobian.runtime.bootstrap import bootstrap_services
from jacobian.runtime.config import CheckerAuthorityMode, RuntimeOptions
from jacobian.runtime.services import (
    ApplicationServices,
    CoreServices,
    build_application_services,
)


@dataclass(frozen=True, slots=True)
class DomainTestServices:
    """A domain test's explicit foundational and application service graphs."""

    core: CoreServices
    application: ApplicationServices
    installation: InstallationContext


@dataclass(frozen=True, slots=True)
class ReferenceTestServices(DomainTestServices):
    """Application services plus explicitly selected production references."""

    references: dict[str, ReferenceInstallation]


@contextmanager
def atomic_installation(core: CoreServices) -> Iterator[None]:
    """Apply the same durable boundary as complete portfolio installation."""

    with (
        core.checkers.policy_transaction(),
        core.store.transaction(),
        cached_package_digests(),
    ):
        yield


@contextmanager
def open_domain_services(
    root: str | Path,
    *bundles: DomainBundle,
    options: RuntimeOptions | None = None,
    checker_authority: CheckerAuthorityMode | None = None,
) -> Iterator[DomainTestServices]:
    """Open core/application services and one production installation context.

    No built-in portfolio is imported or installed here.  A domain test passes
    its literal ``DomainBundle`` to the production domain installer itself.
    """

    if options is not None and checker_authority is not None:
        raise ValueError("pass either options or checker_authority, not both")
    resolved_options = options or RuntimeOptions(
        checker_authority=checker_authority or CheckerAuthorityMode.NONE,
    )
    core = bootstrap_services(root, resolved_options)
    try:
        application = build_application_services(core)
        try:
            installation = create_installation_context(
                core,
                application,
                resolved_options,
            )
            if bundles:
                with atomic_installation(core):
                    DomainBundleInstaller(installation).install(
                        PortfolioPlan(domain_bundles=tuple(bundles))
                    )
            yield DomainTestServices(
                core=core,
                application=application,
                installation=installation,
            )
        finally:
            application.close()
    finally:
        core.close()


@contextmanager
def open_reference_services(
    root: str | Path,
    *reference_names: Literal["erdos_straus", "graph_paths", "matrices"],
    authorize_checkers: bool = False,
) -> Iterator[ReferenceTestServices]:
    """Open only the named production reference-domain installations."""

    authority = (
        CheckerAuthorityMode.INSTALL_BUNDLED
        if authorize_checkers
        else CheckerAuthorityMode.NONE
    )
    with open_domain_services(root, checker_authority=authority) as services:
        references: dict[str, ReferenceInstallation] = {}
        with atomic_installation(services.core):
            for name in reference_names:
                if name == "graph_paths":
                    reference = services.application.reference_installer.install_graph_paths(
                        authorize_checker=services.installation.authorizes_bundled_checkers
                    )
                elif name == "matrices":
                    reference = services.application.reference_installer.install_matrices(
                        authorize_checker=services.installation.authorizes_bundled_checkers
                    )
                else:
                    reference = services.application.reference_installer.install_erdos_straus(
                        authorize_checker=services.installation.authorizes_bundled_checkers
                    )
                references[name] = reference
        yield ReferenceTestServices(
            core=services.core,
            application=services.application,
            installation=services.installation,
            references=references,
        )
