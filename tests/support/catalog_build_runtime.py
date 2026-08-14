"""Test-only owner for complete catalog-build services and adapters."""

from __future__ import annotations

from pathlib import Path

from jacobian.catalog_build import build_catalog_operations
from jacobian.catalog_build_context import create_catalog_build_context
from jacobian.catalog_build_resources import CatalogBuildResources
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.polytope import PolytopeService
from jacobian.runtime.bootstrap import bootstrap_services
from jacobian.runtime.model import JacobianRuntime
from jacobian.runtime.resources import RuntimeResources
from jacobian.verification.service import VerificationService
from tests.support.catalog_build_options import CheckerAuthorityMode


class CatalogBuildRuntime(JacobianRuntime):
    """Expose compiler resources only to tests that exercise their internals."""

    def __init__(
        self,
        core: RuntimeResources,
        verification: VerificationService,
        polytope: PolytopeService,
        resources: CatalogBuildResources,
    ) -> None:
        super().__init__(
            core,
            verification,
            polytope,
        )
        for resource in (
            resources.lean_declarations,
            resources.lean_exploration.repl
            if resources.lean_exploration is not None
            else None,
            resources.lean,
        ):
            if resource is not None:
                core.own(resource)
        self.catalog_build_resources = resources


def create_catalog_build_runtime(
    root: str | Path,
    *,
    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.NONE,
    operation_policy: OperationVisibilityPolicy | None = None,
) -> JacobianRuntime:
    """Build all catalog adapters for integration and provider tests."""

    core = bootstrap_services(
        root,
        operation_policy=operation_policy,
        bind_existing_checkers=(
            checker_authority is CheckerAuthorityMode.HYDRATE_EXISTING
        ),
    )
    try:
        verification = VerificationService(
            core.store,
            core.checkers,
            core.schemas,
            checker_timeout_seconds=105,
        )
        polytope = PolytopeService(core.store, core.schemas)
        context = create_catalog_build_context(
            core,
            verification,
            authorize_bundled_checkers=(
                checker_authority is CheckerAuthorityMode.INSTALL_BUNDLED
            ),
        )
        resources = build_catalog_operations(context, polytope)
        return CatalogBuildRuntime(core, verification, polytope, resources)
    except BaseException as error:
        try:
            core.close()
        except BaseException as cleanup_error:
            error.add_note(
                "catalog-build runtime cleanup also failed: " + str(cleanup_error)
            )
        raise


__all__ = ["create_catalog_build_runtime"]
