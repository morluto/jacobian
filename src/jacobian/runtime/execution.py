"""Construct a catalog-backed runtime for selected operation execution."""

from __future__ import annotations

from pathlib import Path

from jacobian.operation_catalog import OperationCatalog
from jacobian.operation_dispatcher import OperationDispatcher
from jacobian.operation_registry import OperationRegistry
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.polytope import PolytopeService
from jacobian.registry import CheckerRegistry
from jacobian.runtime.bootstrap import bootstrap_services
from jacobian.runtime.model import JacobianRuntime
from jacobian.verification.service import VerificationService


def create_execution_runtime(
    root: str | Path,
    catalog: OperationCatalog,
    *,
    operation_policy: OperationVisibilityPolicy,
    checker_registry: CheckerRegistry | None = None,
) -> JacobianRuntime:
    """Open artifact state and defer implementation loading until selection."""

    core = bootstrap_services(
        root,
        operation_policy=operation_policy,
        bind_existing_checkers=True,
    )
    if checker_registry is not None:
        core.checkers = checker_registry
    verification = VerificationService(
        core.store,
        core.checkers,
        core.schemas,
        checker_timeout_seconds=105,
    )
    polytope = PolytopeService(core.store, core.schemas)
    registry = OperationRegistry(
        catalog,
        core.binder,
        verification,
        core.checkers,
        core.polynomial_expressions,
        polytope,
        core.sat,
        core.smt,
    )
    core.operations = OperationDispatcher(catalog, registry)
    return JacobianRuntime(core, verification, polytope)


__all__ = ["create_execution_runtime"]
