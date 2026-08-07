"""Construction of foundational runtime-owned services."""

from __future__ import annotations

from pathlib import Path

from jacobian.artifacts import ArtifactService
from jacobian.capability_service import CapabilityService
from jacobian.matrices.linear import install_linear_artifacts
from jacobian.matrices.normal_forms import install_matrix_normal_form_artifacts
from jacobian.operation_installation import OperationInstaller
from jacobian.plugins.registry import PluginRegistry
from jacobian.polynomial_expressions import install_polynomial_expression_artifacts
from jacobian.reasoning_log import ReasoningLogService
from jacobian.registry import CheckerRegistry
from jacobian.runtime.config import CheckerAuthorityMode, RuntimeOptions
from jacobian.runtime.services import CoreServices
from jacobian.sat_smt.sat import install_sat_artifacts
from jacobian.sat_smt.smt import install_smt_artifacts
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository


def bootstrap_services(root: str | Path, options: RuntimeOptions) -> CoreServices:
    """Open storage and construct the capability-independent service graph."""

    store = ArtifactRepository(root)
    try:
        schemas = SchemaRegistry(store)
        artifacts = ArtifactService(store, schemas)
        operations = OperationInstaller(store, schemas, artifacts)
        with store.transaction():
            sat = install_sat_artifacts(store, schemas, artifacts)
            smt = install_smt_artifacts(store, schemas, artifacts)
            linear = install_linear_artifacts(store, schemas, artifacts)
            matrix_normal_forms = install_matrix_normal_form_artifacts(
                store,
                schemas,
                artifacts,
            )
            polynomial_expressions = install_polynomial_expression_artifacts(
                store,
                schemas,
                artifacts,
            )
        plugins = PluginRegistry(store, schemas)
        checkers = CheckerRegistry(store)
        checkers.bind_existing_when_omitted = (
            options.checker_authority is CheckerAuthorityMode.HYDRATE_EXISTING
        )
        capabilities = CapabilityService(
            store,
            policy=options.capability_policy,
        )
        reasoning_log = ReasoningLogService(store)
        return CoreServices(
            store=store,
            schemas=schemas,
            artifacts=artifacts,
            operations=operations,
            sat=sat,
            smt=smt,
            linear=linear,
            matrix_normal_forms=matrix_normal_forms,
            polynomial_expressions=polynomial_expressions,
            plugins=plugins,
            checkers=checkers,
            capabilities=capabilities,
            reasoning_log=reasoning_log,
        )
    except BaseException as exc:
        try:
            store.close()
        except BaseException as cleanup_exc:
            exc.add_note(f"service bootstrap cleanup also failed: {cleanup_exc}")
        raise
