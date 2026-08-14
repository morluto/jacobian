"""Construction of foundational runtime-owned resources."""

from __future__ import annotations

from pathlib import Path

from jacobian.artifacts import ArtifactService
from jacobian.catalog.collector import CatalogOperationCollector
from jacobian.operation_binding import OperationBinder
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.polynomial_expressions import (
    PolynomialExpressionArtifactService,
    install_polynomial_expression_artifacts,
)
from jacobian.registry import CheckerRegistry
from jacobian.runtime.resources import RuntimeResources
from jacobian.sat_smt.sat import SatArtifactService, install_sat_artifacts
from jacobian.sat_smt.smt import SmtArtifactService, install_smt_artifacts
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.value_references import ValueReferenceStore


def bootstrap_services(
    root: str | Path,
    *,
    operation_policy: OperationVisibilityPolicy | None = None,
    bind_existing_checkers: bool = False,
    install_family_artifacts: bool = True,
    collect_operations: bool = True,
) -> RuntimeResources:
    """Open storage and construct operation-independent resources.

    Serving execution must pass ``collect_operations=False``. Catalog
    compilation and domain-test inventory keep the collector.
    """

    store = ArtifactRepository(root)
    try:
        schemas = SchemaRegistry(store)
        artifacts = ArtifactService(store, schemas)
        values = ValueReferenceStore()
        binder = OperationBinder(store, schemas, artifacts, values)
        sat: SatArtifactService | None = None
        smt: SmtArtifactService | None = None
        polynomial_expressions: PolynomialExpressionArtifactService | None = None
        if install_family_artifacts:
            with store.transaction():
                sat = install_sat_artifacts(store, schemas, artifacts)
                smt = install_smt_artifacts(store, schemas, artifacts)
                polynomial_expressions = install_polynomial_expression_artifacts(
                    store,
                    schemas,
                    artifacts,
                )
        checkers = CheckerRegistry(store)
        checkers.bind_existing_when_omitted = bind_existing_checkers
        operations = (
            CatalogOperationCollector(
                store,
                policy=operation_policy,
            )
            if collect_operations
            else None
        )
        return RuntimeResources(
            store=store,
            schemas=schemas,
            artifacts=artifacts,
            values=values,
            binder=binder,
            checkers=checkers,
            operations=operations,
            sat=sat,
            smt=smt,
            polynomial_expressions=polynomial_expressions,
        )
    except BaseException as exc:
        try:
            store.close()
        except BaseException as cleanup_exc:
            exc.add_note(f"service bootstrap cleanup also failed: {cleanup_exc}")
        raise
