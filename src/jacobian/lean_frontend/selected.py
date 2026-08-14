"""Resolve one selected built-in Lean operation from compiled state."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jacobian.builtin_operations import LeanCheckAdapter
from jacobian.checker_authorization import register_lean_checker_contracts
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.operations import OperationDescriptor
from jacobian.lean_frontend.declaration_operations import (
    lean_declaration_query_operations,
)
from jacobian.lean_frontend.declarations import installed_lean_declaration_service
from jacobian.lean_frontend.exploration import install_lean_exploration_operations
from jacobian.lean_frontend.proof_axioms import install_lean_proof_axioms_operation
from jacobian.lean_frontend.proof_edit import install_lean_proof_edit_operation
from jacobian.lean_frontend.proof_state_inspect import (
    install_lean_proof_state_inspect_only,
)
from jacobian.lean_frontend.service import LeanService
from jacobian.lean_frontend.statement import install_lean_statement_operations
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_binding import OperationBinder
from jacobian.operation_catalog import OperationCatalog, OperationCatalogError
from jacobian.providers.lean_runtime import lean_provider_runtime
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService

_STATEMENT_OPERATIONS = {
    "lean.statement.propose",
    "lean.statement.compare",
}
_DECLARATION_OPERATIONS = {
    "lean.declaration.search",
    "lean.declaration.inspect",
    "lean.declaration.dependencies",
}
_EXPLORATION_OPERATIONS = {
    "lean.proof_state.apply_tactic",
    "lean.retrieve.premises",
    "lean.term.apply",
    "lean.proof_state.inspect",
    "lean.proof_state.metavariable_fields",
}


def bind_selected_lean_operation(
    operation_id: str,
    descriptor: OperationDescriptor,
    catalog: OperationCatalog,
    binder: OperationBinder,
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    verification: VerificationService,
    checkers: CheckerRegistry,
    own: Callable[[object], None],
) -> OperationAdapter[Any] | None:
    """Construct only the Lean service family selected by ``operation_id``."""

    installations = _lean_installations(catalog, store, schemas, checkers)
    profiles = {
        environment.value: {
            "semantics_uri": installation.semantics_uri,
            "lean_version": installation.lean_version,
            "lean_commit": installation.lean_commit,
            "import_name": installation.import_name,
            "mathlib_commit": installation.mathlib_commit,
            "allowed_axioms": list(installation.allowed_axioms),
            "checker_timeout_seconds": installation.checker_timeout_seconds,
        }
        for environment, installation in installations.items()
    }
    checker_ids = tuple(
        installation.checker_id
        for installation in installations.values()
        if installation.checker_id is not None
    )
    runtime = lean_provider_runtime(profiles=profiles, checker_ids=checker_ids)
    if operation_id in _STATEMENT_OPERATIONS:
        statement_adapters, _ = install_lean_statement_operations(
            store,
            schemas,
            binder.artifacts,
            runtime,
        )
        return _select(statement_adapters, operation_id)
    if operation_id in _DECLARATION_OPERATIONS:
        declarations = installed_lean_declaration_service(
            runtime,
            cache_root=store.root / "cache" / "lean-declarations",
        )
        own(declarations)
        return _select(
            binder.bind(lean_declaration_query_operations(declarations)).adapters,
            operation_id,
        )

    if operation_id == "lean.proof_state.inspect":
        return install_lean_proof_state_inspect_only(
            store,
            schemas,
            binder.artifacts,
            installations,
            runtime,
        )
    if operation_id in _EXPLORATION_OPERATIONS:
        exploration_adapters, exploration = install_lean_exploration_operations(
            store,
            schemas,
            binder.artifacts,
            installations,
            runtime,
        )
        own(exploration.repl)
        return _select(exploration_adapters, operation_id)
    if operation_id == "lean.proof.axioms.inspect":
        axioms_adapter, _ = install_lean_proof_axioms_operation(
            store,
            schemas,
            binder.artifacts,
            installations,
            runtime,
        )
        return axioms_adapter
    if operation_id in {"lean.check", "lean.proof_edit.validate"}:
        lean = LeanService(
            store,
            binder.artifacts,
            verification,
            installations,
        )
        own(lean)
        if operation_id == "lean.check":
            return LeanCheckAdapter(lean, runtime)
        proof_edit_adapter, _ = install_lean_proof_edit_operation(
            store,
            schemas,
            binder.artifacts,
            lean,
            runtime,
        )
        return proof_edit_adapter
    return None


def _lean_installations(
    catalog: OperationCatalog,
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    checkers: CheckerRegistry,
) -> dict[LeanEnvironment, Any]:
    checker_ids = tuple(
        binding.checker_id for binding in catalog.checker_bindings("lean.check")
    )
    if checker_ids and len(checker_ids) != 2:
        raise OperationCatalogError(
            "Lean checker inventory is stale; run `jacobian update`"
        )
    selected = {
        environment: (checker_ids[index] if checker_ids else None)
        for index, environment in enumerate(
            (LeanEnvironment.CORE, LeanEnvironment.MATHLIB)
        )
    }
    for checker_id in checker_ids:
        checkers.require_active(str(checker_id))
    installations, _ = register_lean_checker_contracts(
        store,
        schemas,
        checker_ids=selected,
    )
    return installations


def _select(
    adapters: tuple[OperationAdapter[Any], ...],
    operation_id: str,
) -> OperationAdapter[Any]:
    return next(
        adapter
        for adapter in adapters
        if adapter.descriptor.operation_id == operation_id
    )


__all__ = ["bind_selected_lean_operation"]
