"""Construction of foundational runtime-owned services."""

from __future__ import annotations

from pathlib import Path

from jacobian.artifacts import ArtifactService
from jacobian.operation_installation import OperationInstaller
from jacobian.operation_service import OperationService
from jacobian.polynomial_expressions import install_polynomial_expression_artifacts
from jacobian.registry import CheckerRegistry
from jacobian.runtime.config import CheckerAuthorityMode, RuntimeOptions
from jacobian.runtime.services import CoreServices
from jacobian.sat_smt.sat import install_sat_artifacts
from jacobian.sat_smt.smt import install_smt_artifacts
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.value_references import ValueReferenceStore


def bootstrap_services(root: str | Path, options: RuntimeOptions) -> CoreServices:
    """Open storage and construct the operation-independent service graph."""

    store = ArtifactRepository(root)
    try:
        schemas = SchemaRegistry(store)
        artifacts = ArtifactService(store, schemas)
        values = ValueReferenceStore()
        installer = OperationInstaller(store, schemas, artifacts, values)
        with store.transaction():
            sat = install_sat_artifacts(store, schemas, artifacts)
            smt = install_smt_artifacts(store, schemas, artifacts)
            polynomial_expressions = install_polynomial_expression_artifacts(
                store,
                schemas,
                artifacts,
            )
        checkers = CheckerRegistry(store)
        checkers.bind_existing_when_omitted = (
            options.checker_authority is CheckerAuthorityMode.HYDRATE_EXISTING
        )
        operations = OperationService(
            store,
            policy=options.operation_policy,
        )
        return CoreServices(
            store=store,
            schemas=schemas,
            artifacts=artifacts,
            values=values,
            installer=installer,
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
