"""Shared harness for installing exact-domain checkers in domain tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from jacobian.builtin_operation_modules import load_builtin_operation_modules
from jacobian.catalog.build import CatalogOperationBuilder
from jacobian.operation_binding import BoundOperationGroup
from jacobian.operation_declarations import OperationDeclarations
from tests.support.catalog_build_options import CheckerAuthorityMode
from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)


@dataclass(frozen=True, slots=True)
class VerifiedDomainTestServices(DomainTestServices):
    """Focused services plus the exact installed bundle resources."""

    operation_groups: dict[str, BoundOperationGroup]


def install_verified_domain_bundles(
    services: DomainTestServices,
    *operation_groups: OperationDeclarations,
) -> dict[str, BoundOperationGroup]:
    """Install selected bundles through the production portfolio path."""

    if not operation_groups:
        raise ValueError("at least one verified operation group is required")
    builtin = {
        tuple(operation.operation_id for operation in operations): (
            module_name,
            checker_declarations,
        )
        for module_name, operations, checker_declarations in (
            load_builtin_operation_modules()
        )
    }
    bound_by_name: dict[str, BoundOperationGroup] = {}
    exact_groups = {}
    with atomic_installation(services.core):
        for operations in operation_groups:
            operation_ids = tuple(operation.operation_id for operation in operations)
            module_name, checker_declarations = builtin[operation_ids]
            bound = services.installation.binder.bind(operations)
            for adapter in bound.adapters:
                services.installation.register_operation(adapter)
            name = operation_ids[0].split(".", maxsplit=1)[0]
            bound_by_name[name] = bound
            exact_groups[module_name] = operations, bound, checker_declarations
        CatalogOperationBuilder(services.installation).bind_domain_verification(
            exact_groups
        )
    return bound_by_name


@contextmanager
def open_exact_domain_services(
    root: str | Path,
    *operation_groups: OperationDeclarations,
    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.INSTALL_BUNDLED,
) -> Iterator[VerifiedDomainTestServices]:
    """Open domain services with explicitly selected verified operations.

    Ordinary domain fixtures declare their operations rather than assembling the
    checker and adapter registration recipe.
    """

    with open_domain_services(root, checker_authority=checker_authority) as services:
        installed = install_verified_domain_bundles(services, *operation_groups)
        yield VerifiedDomainTestServices(
            core=services.core,
            verification=services.verification,
            polytope=services.polytope,
            installation=services.installation,
            operation_groups=installed,
        )
