"""Operator-owned dependencies used while compiling the built-in catalog."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jacobian.artifacts import ArtifactService
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_binding import OperationBinder
from jacobian.polynomial_expressions import PolynomialExpressionArtifactService
from jacobian.registry import CheckerRegistry
from jacobian.sat_smt.sat import SatArtifactService
from jacobian.sat_smt.smt import SmtArtifactService
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.value_references import ValueReferenceStore
from jacobian.verification.service import VerificationService

if TYPE_CHECKING:
    from jacobian.runtime.resources import RuntimeResources


@dataclass(frozen=True, slots=True)
class CatalogBuildContext:
    """Resources required while compiling the explicit built-in catalog."""

    store: ArtifactRepository
    schemas: SchemaRegistry
    artifacts: ArtifactService
    values: ValueReferenceStore
    checkers: CheckerRegistry
    verification: VerificationService
    binder: OperationBinder
    sat: SatArtifactService
    smt: SmtArtifactService
    polynomial_expressions: PolynomialExpressionArtifactService
    authorize_bundled_checkers: bool
    register_operation: Callable[[OperationAdapter[Any]], None]


def create_catalog_build_context(
    core: RuntimeResources,
    verification: VerificationService,
    *,
    authorize_bundled_checkers: bool = False,
) -> CatalogBuildContext:
    """Build the operator-only context for one catalog compilation.

    The context is derived from explicit core and runtime ownership. It keeps
    operation visibility and operator-owned checker authority at the one
    catalog build boundary.

    The registrar is the only place where operation exclusions are applied;
    every built-in adapter therefore follows the same policy.
    """

    def register(adapter: OperationAdapter[Any]) -> None:
        core.operations.register(adapter)

    return CatalogBuildContext(
        store=core.store,
        schemas=core.schemas,
        artifacts=core.artifacts,
        values=core.values,
        checkers=core.checkers,
        verification=verification,
        binder=core.binder,
        sat=core.sat,
        smt=core.smt,
        polynomial_expressions=core.polynomial_expressions,
        authorize_bundled_checkers=authorize_bundled_checkers,
        register_operation=register,
    )
