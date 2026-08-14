"""Construction of foundational runtime-owned resources."""

from __future__ import annotations

from pathlib import Path

from jacobian.artifacts import ArtifactService
from jacobian.catalog_operation_collector import CatalogOperationCollector
from jacobian.operation_binding import OperationBinder
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.polynomial_expressions import install_polynomial_expression_artifacts
from jacobian.registry import CheckerRegistry
from jacobian.runtime.resources import RuntimeResources
from jacobian.sat_smt.sat import install_sat_artifacts
from jacobian.sat_smt.smt import install_smt_artifacts
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.value_references import ValueReferenceStore


def bootstrap_services(
    root: str | Path,
    *,
    operation_policy: OperationVisibilityPolicy | None = None,
    bind_existing_checkers: bool = False,
) -> RuntimeResources:
    """Open storage and construct operation-independent resources."""

    store = ArtifactRepository(root)
    try:
        schemas = SchemaRegistry(store)
        artifacts = ArtifactService(store, schemas)
        values = ValueReferenceStore()
        binder = OperationBinder(store, schemas, artifacts, values)
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
        operations = CatalogOperationCollector(
            store,
            policy=operation_policy,
        )
        return RuntimeResources(
            store=store,
            schemas=schemas,
            artifacts=artifacts,
            values=values,
            binder=binder,
            sat=sat,
            smt=smt,
            polynomial_expressions=polynomial_expressions,
            checkers=checkers,
            operations=operations,
        )
    except BaseException as exc:
        try:
            store.close()
        except BaseException as cleanup_exc:
            exc.add_note(f"service bootstrap cleanup also failed: {cleanup_exc}")
        raise
