"""Lazy binding for retained resource-backed operations outside domain families."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jacobian.contracts.operations import OperationDescriptor
from jacobian.operation_binding import OperationBinder
from jacobian.operation_catalog import OperationCatalog
from jacobian.polytope import PolytopeService
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry
from jacobian.selected_operation_bindings import SelectedOperationBinding
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService

if TYPE_CHECKING:
    from jacobian.catalog.build import CatalogBuildContext

SELECTED_CORE_OPERATION_IDS = frozenset(
    {
        "polytope.separate",
        "finite.coverage.verify",
        "finite_magma.table.enumerate",
        "universal_algebra.evaluate_laws",
        "universal_algebra.search.countermodel",
        "universal_algebra.law_evaluation.verify",
    }
)


def bind_selected_core_operation(
    operation_id: str,
    descriptor: OperationDescriptor,
    *,
    binder: OperationBinder,
    verification: VerificationService,
    checkers: CheckerRegistry,
    catalog: OperationCatalog,
    polytope: PolytopeService,
    store: ArtifactRepository,
    schemas: SchemaRegistry,
) -> SelectedOperationBinding | None:
    """Bind one retained resource-backed operation from runtime services."""

    if operation_id not in SELECTED_CORE_OPERATION_IDS:
        return None
    if operation_id == "polytope.separate":
        from jacobian.polytope_operations import PolytopeSeparationAdapter

        return SelectedOperationBinding(PolytopeSeparationAdapter(polytope))
    if operation_id == "finite.coverage.verify":
        from jacobian.finite_coverage import bind_selected_finite_coverage

        return SelectedOperationBinding(
            bind_selected_finite_coverage(
                store,
                schemas,
                binder.artifacts,
                verification,
                checkers,
                catalog,
            )
        )
    from jacobian.universal_algebra_operations import (
        bind_selected_universal_algebra_operation,
    )

    adapter = bind_selected_universal_algebra_operation(
        operation_id,
        descriptor,
        store,
        schemas,
        binder.artifacts,
        verification,
        checkers,
        catalog,
    )
    if adapter is None:
        return None
    return SelectedOperationBinding(adapter)


def install_selected_core_catalog(
    context: CatalogBuildContext,
    *,
    polytope: object | None = None,
    resources: object | None = None,
) -> None:
    """Compile polytope, finite-coverage, and universal-algebra operations."""

    del resources
    from jacobian.checker_authorization import install_polytope_checkers
    from jacobian.finite_coverage import install_finite_coverage
    from jacobian.polytope_operations import PolytopeSeparationAdapter
    from jacobian.universal_algebra_operations import (
        install_universal_algebra_operations,
    )

    if not isinstance(polytope, PolytopeService):
        raise TypeError("core catalog install requires a polytope service")
    ctx = context
    ctx.register_operation(PolytopeSeparationAdapter(polytope))
    finite_coverage_adapter, _ = install_finite_coverage(
        ctx.store,
        ctx.schemas,
        ctx.artifacts,
        ctx.verification,
        ctx.checkers,
        authorize_checker=ctx.authorize_bundled_checkers,
    )
    if finite_coverage_adapter is not None:
        ctx.register_operation(finite_coverage_adapter)
    universal_adapters, _ = install_universal_algebra_operations(
        ctx.store,
        ctx.schemas,
        ctx.artifacts,
        ctx.verification,
        ctx.checkers,
        authorize_checker=ctx.authorize_bundled_checkers,
    )
    for universal_adapter in universal_adapters:
        ctx.register_operation(universal_adapter)
    if ctx.authorize_bundled_checkers or ctx.checkers.bind_existing_when_omitted:
        install_polytope_checkers(
            ctx.checkers,
            claim_schema_uri=polytope.claim_schema_uri,
            semantics_uri=polytope.semantics_uri,
            point_schema_uri=polytope.point_schema_uri,
        )


__all__ = [
    "SELECTED_CORE_OPERATION_IDS",
    "bind_selected_core_operation",
    "install_selected_core_catalog",
]
