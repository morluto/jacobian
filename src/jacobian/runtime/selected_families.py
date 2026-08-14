"""The fixed selected-operation family table for one execution runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from jacobian.contracts.operations import OperationDescriptor
from jacobian.graphs.operation_resources import (
    SELECTED_GRAPH_OPERATION_IDS,
    bind_selected_graph_operation,
    install_selected_graph_catalog,
)
from jacobian.lean_frontend.selected import (
    SELECTED_LEAN_OPERATION_IDS,
    bind_selected_lean_operation,
    install_selected_lean_catalog,
)
from jacobian.operation_binding import OperationBinder
from jacobian.operation_catalog import OperationCatalog
from jacobian.polynomial_expressions import PolynomialExpressionArtifactService
from jacobian.polynomials.selected import (
    SELECTED_POLYNOMIAL_OPERATION_IDS,
    bind_selected_polynomial_operation,
    install_selected_polynomial_catalog,
)
from jacobian.polytope import PolytopeService
from jacobian.registry import CheckerRegistry
from jacobian.sat_smt.sat import SatArtifactService
from jacobian.sat_smt.selected import (
    SELECTED_SAT_SMT_OPERATION_IDS,
    bind_selected_sat_smt_operation,
    install_selected_sat_smt_catalog,
)
from jacobian.sat_smt.smt import SmtArtifactService
from jacobian.selected_operation_bindings import (
    RuntimeSelectedFamily,
    SelectedFamilySpec,
    SelectedOperationBinding,
)
from jacobian.selected_operations import (
    SELECTED_CORE_OPERATION_IDS,
    bind_selected_core_operation,
    install_selected_core_catalog,
)
from jacobian.verification.service import VerificationService

_FAMILY_SPECS = (
    SelectedFamilySpec("family:graph", SELECTED_GRAPH_OPERATION_IDS),
    SelectedFamilySpec("family:polynomial", SELECTED_POLYNOMIAL_OPERATION_IDS),
    SelectedFamilySpec("family:lean", SELECTED_LEAN_OPERATION_IDS),
    SelectedFamilySpec("family:sat-smt", SELECTED_SAT_SMT_OPERATION_IDS),
    SelectedFamilySpec("family:core", SELECTED_CORE_OPERATION_IDS),
)

_FAMILY_CATALOG_INSTALLERS: dict[str, Callable[..., None]] = {
    "family:graph": install_selected_graph_catalog,
    "family:polynomial": install_selected_polynomial_catalog,
    "family:lean": install_selected_lean_catalog,
    "family:sat-smt": install_selected_sat_smt_catalog,
    "family:core": install_selected_core_catalog,
}


def selected_family_specs() -> tuple[SelectedFamilySpec, ...]:
    """Return immutable family ownership metadata used by catalog builds."""

    return _FAMILY_SPECS


def selected_family_catalog_installers() -> Mapping[str, Callable[..., None]]:
    """Return the 1:1 compile hooks indexed by ``selected_family_specs()``."""

    expected = tuple(spec.origin for spec in _FAMILY_SPECS)
    actual = tuple(_FAMILY_CATALOG_INSTALLERS)
    if actual != expected:
        raise RuntimeError(
            "selected family catalog installers must match selected_family_specs: "
            f"expected={list(expected)} actual={list(actual)}"
        )
    return _FAMILY_CATALOG_INSTALLERS


def selected_operation_origin(operation_id: str) -> str | None:
    """Return the persisted family origin for one resource-backed operation."""

    for spec in _FAMILY_SPECS:
        if operation_id in spec.operation_ids:
            return spec.origin
    return None


def create_runtime_selected_families(
    *,
    catalog: OperationCatalog,
    binder: OperationBinder,
    verification: VerificationService,
    checkers: CheckerRegistry,
    polynomial_expressions: PolynomialExpressionArtifactService,
    polytope: PolytopeService,
    sat: SatArtifactService,
    smt: SmtArtifactService,
    runtime_resources: Any,
) -> tuple[RuntimeSelectedFamily, ...]:
    """Capture all selected binders in one fixed, runtime-local table."""

    store = runtime_resources.store
    schemas = runtime_resources.schemas
    artifacts = runtime_resources.artifacts

    def graph(
        operation_id: str, _descriptor: OperationDescriptor
    ) -> SelectedOperationBinding | None:
        adapter = bind_selected_graph_operation(
            operation_id,
            store,
            schemas,
            artifacts,
            verification,
            checkers,
            catalog,
        )
        return None if adapter is None else SelectedOperationBinding(adapter)

    def polynomial(
        operation_id: str, descriptor: OperationDescriptor
    ) -> SelectedOperationBinding | None:
        return bind_selected_polynomial_operation(
            operation_id,
            descriptor,
            binder=binder,
            verification=verification,
            checkers=checkers,
            polynomial_expressions=polynomial_expressions,
            catalog=catalog,
            store=store,
            schemas=schemas,
        )

    def lean(
        operation_id: str, descriptor: OperationDescriptor
    ) -> SelectedOperationBinding | None:
        return bind_selected_lean_operation(
            operation_id,
            descriptor,
            catalog,
            binder,
            store,
            schemas,
            verification,
            checkers,
        )

    def sat_smt(
        operation_id: str, descriptor: OperationDescriptor
    ) -> SelectedOperationBinding | None:
        return bind_selected_sat_smt_operation(
            operation_id,
            descriptor,
            binder=binder,
            verification=verification,
            checkers=checkers,
            catalog=catalog,
            store=store,
            schemas=schemas,
            sat=sat,
            smt=smt,
        )

    def core(
        operation_id: str, descriptor: OperationDescriptor
    ) -> SelectedOperationBinding | None:
        return bind_selected_core_operation(
            operation_id,
            descriptor,
            binder=binder,
            verification=verification,
            checkers=checkers,
            catalog=catalog,
            polytope=polytope,
            store=store,
            schemas=schemas,
        )

    return (
        RuntimeSelectedFamily(_FAMILY_SPECS[0], graph),
        RuntimeSelectedFamily(_FAMILY_SPECS[1], polynomial),
        RuntimeSelectedFamily(_FAMILY_SPECS[2], lean),
        RuntimeSelectedFamily(_FAMILY_SPECS[3], sat_smt),
        RuntimeSelectedFamily(_FAMILY_SPECS[4], core),
    )


__all__ = [
    "create_runtime_selected_families",
    "selected_family_catalog_installers",
    "selected_family_specs",
    "selected_operation_origin",
]
