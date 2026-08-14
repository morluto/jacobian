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
from jacobian.runtime.selected_families import create_runtime_selected_families
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
    try:
        if checker_registry is not None:
            core.checkers = checker_registry
        verification = VerificationService(
            core.store,
            core.checkers,
            core.schemas,
            checker_timeout_seconds=105,
        )
        polytope = PolytopeService(core.store, core.schemas)
        selected_families = create_runtime_selected_families(
            catalog=catalog,
            binder=core.binder,
            verification=verification,
            checkers=core.checkers,
            polynomial_expressions=core.polynomial_expressions,
            polytope=polytope,
            sat=core.sat,
            smt=core.smt,
            runtime_resources=core,
        )
        registry = OperationRegistry(
            catalog,
            core.binder,
            verification,
            core.checkers,
            selected_families,
            core,
        )
        core.operations = OperationDispatcher(catalog, registry)
        return JacobianRuntime(core, verification, polytope)
    except BaseException as exc:
        try:
            core.close()
        except BaseException as cleanup_exc:
            exc.add_note(f"runtime construction cleanup also failed: {cleanup_exc}")
        raise


__all__ = ["create_execution_runtime"]
