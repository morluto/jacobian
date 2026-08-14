"""Lazy binding for retained resource-backed operations outside domain families."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDescriptor
from jacobian.operation_binding import OperationBinder
from jacobian.operation_catalog import OperationCatalog
from jacobian.polytope import PolytopeService
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry
from jacobian.selected_operation_bindings import SelectedOperationBinding
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService

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


__all__ = ["SELECTED_CORE_OPERATION_IDS", "bind_selected_core_operation"]
