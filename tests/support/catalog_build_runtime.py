"""Test-only owner for complete catalog-build services and adapters."""

from __future__ import annotations

from pathlib import Path

from jacobian.catalog_build import build_catalog_operations
from jacobian.catalog_build_context import create_catalog_build_context
from jacobian.catalog_build_resources import CatalogBuildResources
from jacobian.operation_service import OperationPolicy
from jacobian.runtime import CheckerAuthorityMode
from jacobian.runtime.bootstrap import bootstrap_services
from jacobian.runtime.config import RuntimeOptions
from jacobian.runtime.model import JacobianRuntime
from jacobian.runtime.services import (
    CoreServices,
    RuntimeServices,
    build_runtime_services,
)


class CatalogBuildRuntime(JacobianRuntime):
    """Expose compiler resources only to tests that exercise their internals."""

    def __init__(
        self,
        core: CoreServices,
        services: RuntimeServices,
        resources: CatalogBuildResources,
    ) -> None:
        super().__init__(
            core,
            services,
            close_resources=resources.close,
            start_lean_warmup=lambda: _start_lean_warmup(resources),
        )
        self.catalog_build_resources = resources


def create_catalog_build_runtime(
    root: str | Path,
    *,
    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.NONE,
    operation_exclusions: frozenset[str] = frozenset(),
    operation_policy: OperationPolicy | None = None,
) -> JacobianRuntime:
    """Build all catalog adapters for integration and provider tests."""

    options = RuntimeOptions(
        checker_authority=checker_authority,
        operation_exclusions=operation_exclusions,
        operation_policy=operation_policy,
    )
    core = bootstrap_services(root, options)
    try:
        services = build_runtime_services(core)
        context = create_catalog_build_context(core, services, options)
        resources = build_catalog_operations(context, services)
        return CatalogBuildRuntime(core, services, resources)
    except BaseException as error:
        try:
            core.close()
        except BaseException as cleanup_error:
            error.add_note(
                "catalog-build runtime cleanup also failed: " + str(cleanup_error)
            )
        raise


def _start_lean_warmup(resources: CatalogBuildResources) -> None:
    if resources.lean is not None:
        resources.lean.start_mathlib_warmup()


__all__ = ["create_catalog_build_runtime"]
