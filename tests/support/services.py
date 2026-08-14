"""Tier-local service graph helpers backed by production composition seams."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from jacobian.implementation import cached_package_digests
from jacobian.operation_declarations import OperationDeclarations
from jacobian.portfolio.context import (
    PortfolioContext,
    create_portfolio_context,
)
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
    installation: PortfolioContext


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
    *operation_groups: OperationDeclarations,
    options: RuntimeOptions | None = None,
    checker_authority: CheckerAuthorityMode | None = None,
) -> Iterator[DomainTestServices]:
    """Open core/application services and one production installation context.

    No built-in portfolio is imported or installed here.  A domain test passes
    its literal portfolio component to the production installer itself.
    """

    if options is not None and checker_authority is not None:
        raise ValueError("pass either options or checker_authority, not both")
    resolved_options = options or RuntimeOptions(
        checker_authority=checker_authority or CheckerAuthorityMode.NONE,
    )
    core = bootstrap_services(root, resolved_options)
    try:
        application = build_runtime_services(core)
        installation = create_portfolio_context(
            core,
            application,
            resolved_options,
        )
        if operation_groups:
            with atomic_installation(core):
                for operations in operation_groups:
                    bound = installation.binder.bind(operations)
                    for adapter in bound.adapters:
                        installation.register_operation(adapter)
        yield DomainTestServices(
            core=core,
            application=application,
            installation=installation,
        )
    finally:
        core.close()
