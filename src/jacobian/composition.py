"""Composition root for the runtime service graph and built-in portfolio."""

from __future__ import annotations

from pathlib import Path

from jacobian.catalog_build import build_catalog_operations
from jacobian.catalog_build_context import create_catalog_build_context
from jacobian.catalog_build_resources import CatalogBuildResources
from jacobian.runtime.bootstrap import bootstrap_services
from jacobian.runtime.config import RuntimeOptions
from jacobian.runtime.model import JacobianRuntime
from jacobian.runtime.services import (
    CoreServices,
    RuntimeServices,
    build_runtime_services,
)


class CatalogBuildRuntime(JacobianRuntime):
    """Compatibility owner for compiler-focused tests pending migration."""

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


def compose_runtime(root: str | Path, options: RuntimeOptions) -> JacobianRuntime:
    """Construct one owned runtime and close partial state on failure."""

    core = bootstrap_services(root, options)
    try:
        services = build_runtime_services(core)
        portfolio = create_catalog_build_context(core, services, options)
        catalog_build_resources = build_catalog_operations(portfolio, services)
        return CatalogBuildRuntime(core, services, catalog_build_resources)
    except BaseException as error:
        cleanup_failures: list[BaseException] = []
        try:
            core.close()
        except BaseException as cleanup_error:
            cleanup_failures.append(cleanup_error)
        if cleanup_failures:
            error.add_note(
                "runtime construction cleanup also failed: "
                + "; ".join(str(failure) for failure in cleanup_failures)
            )
        raise


def _start_lean_warmup(resources: CatalogBuildResources) -> None:
    if resources.lean is not None:
        resources.lean.start_mathlib_warmup()


__all__ = ["compose_runtime"]
