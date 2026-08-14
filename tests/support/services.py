"""Tier-local resource helpers backed by production composition seams."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from jacobian.catalog.build import (
    CatalogBuildContext,
    create_catalog_build_context,
)
from jacobian.implementation import cached_package_digests
from jacobian.operation_declarations import OperationDeclarations
from jacobian.polytope import PolytopeService
from jacobian.runtime.bootstrap import bootstrap_services
from jacobian.runtime.resources import RuntimeResources
from jacobian.verification.service import VerificationService
from tests.support.catalog_build_options import CheckerAuthorityMode


@dataclass(frozen=True, slots=True)
class DomainTestServices:
    """Resources owned by one focused domain test."""

    core: RuntimeResources
    verification: VerificationService
    polytope: PolytopeService
    installation: CatalogBuildContext


@contextmanager
def atomic_installation(core: RuntimeResources) -> Iterator[None]:
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
    checker_authority: CheckerAuthorityMode | None = None,
) -> Iterator[DomainTestServices]:
    """Open runtime resources and one production installation context.

    No built-in portfolio is imported or installed here.  A domain test passes
    its literal portfolio component to the production installer itself.
    """

    authority = checker_authority or CheckerAuthorityMode.NONE
    core = bootstrap_services(
        root,
        bind_existing_checkers=(authority is CheckerAuthorityMode.HYDRATE_EXISTING),
    )
    try:
        verification = VerificationService(
            core.store,
            core.checkers,
            core.schemas,
            checker_timeout_seconds=105,
        )
        polytope = PolytopeService(core.store, core.schemas)
        installation = create_catalog_build_context(
            core,
            verification,
            authorize_bundled_checkers=(
                authority is CheckerAuthorityMode.INSTALL_BUNDLED
            ),
        )
        if operation_groups:
            with atomic_installation(core):
                for operations in operation_groups:
                    bound = installation.binder.bind(operations)
                    for adapter in bound.adapters:
                        installation.register_operation(adapter)
        yield DomainTestServices(
            core=core,
            verification=verification,
            polytope=polytope,
            installation=installation,
        )
    finally:
        core.close()
