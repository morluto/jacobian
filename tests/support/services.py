"""Tier-local service graph helpers backed by production composition seams."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from jacobian.implementation import cached_package_digests
from jacobian.installation.context import (
    InstallationContext,
    create_installation_context,
)
from jacobian.operations import DomainBundle
from jacobian.portfolio.domain_installation import DomainBundleInstaller
from jacobian.portfolio.model import PortfolioPlan
from jacobian.runtime.bootstrap import bootstrap_services
from jacobian.runtime.config import CheckerAuthorityMode, RuntimeOptions
from jacobian.runtime.services import (
    CoreServices,
    RuntimeServices,
    build_runtime_services,
)


@dataclass(frozen=True, slots=True)
class DomainTestServices:
    """A domain test's explicit foundational and application service graphs."""

    core: CoreServices
    application: RuntimeServices
    installation: InstallationContext


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
        application = build_runtime_services(core)
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
