"""Resolve one selected built-in Lean operation from compiled state."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

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
from jacobian.providers.lean_runtime import (
    lean_frontend_provider_runtime,
    lean_provider_runtime,
)
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry
from jacobian.selected_operation_bindings import SelectedOperationBinding
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService

if TYPE_CHECKING:
    from jacobian.catalog_build_context import CatalogBuildContext
    from jacobian.catalog_build_resources import CatalogBuildResources

_LOGGER = logging.getLogger(__name__)

_STATEMENT_OPERATIONS = frozenset(
    {
        "lean.statement.propose",
        "lean.statement.compare",
    }
)
_DECLARATION_OPERATIONS = frozenset(
    {
        "lean.declaration.search",
        "lean.declaration.inspect",
        "lean.declaration.dependencies",
    }
)
_EXPLORATION_OPERATIONS = frozenset(
    {
        "lean.proof_state.apply_tactic",
        "lean.retrieve.premises",
        "lean.term.apply",
        "lean.proof_state.inspect",
        "lean.proof_state.metavariable_fields",
    }
)
SELECTED_LEAN_OPERATION_IDS = frozenset(
    {
        "lean.check",
        "lean.declaration.dependencies",
        "lean.declaration.inspect",
        "lean.declaration.search",
        "lean.proof.axioms.inspect",
        "lean.proof_edit.validate",
        "lean.proof_state.apply_tactic",
        "lean.proof_state.inspect",
        "lean.proof_state.metavariable_fields",
        "lean.retrieve.premises",
        "lean.statement.compare",
        "lean.statement.propose",
        "lean.term.apply",
    }
)


def bind_selected_lean_operation(
    operation_id: str,
    descriptor: OperationDescriptor,
    catalog: OperationCatalog,
    binder: OperationBinder,
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    verification: VerificationService,
    checkers: CheckerRegistry,
) -> SelectedOperationBinding | None:
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
    resources: list[object] = []
    if operation_id in _STATEMENT_OPERATIONS:
        statement_adapters, _ = install_lean_statement_operations(
            store,
            schemas,
            binder.artifacts,
            runtime,
        )
        return SelectedOperationBinding(_select(statement_adapters, operation_id))
    if operation_id in _DECLARATION_OPERATIONS:
        declarations = installed_lean_declaration_service(
            runtime,
            cache_root=store.root / "cache" / "lean-declarations",
        )
        resources.append(declarations)
        return SelectedOperationBinding(
            _select(
                binder.bind(lean_declaration_query_operations(declarations)).adapters,
                operation_id,
            ),
            tuple(resources),
        )

    if operation_id == "lean.proof_state.inspect":
        return SelectedOperationBinding(
            install_lean_proof_state_inspect_only(
                store,
                schemas,
                binder.artifacts,
                installations,
                runtime,
            )
        )
    if operation_id in _EXPLORATION_OPERATIONS:
        exploration_adapters, exploration = install_lean_exploration_operations(
            store,
            schemas,
            binder.artifacts,
            installations,
            runtime,
        )
        resources.append(exploration.repl)
        return SelectedOperationBinding(
            _select(exploration_adapters, operation_id), tuple(resources)
        )
    if operation_id == "lean.proof.axioms.inspect":
        axioms_adapter, _ = install_lean_proof_axioms_operation(
            store,
            schemas,
            binder.artifacts,
            installations,
            runtime,
        )
        return SelectedOperationBinding(axioms_adapter)
    if operation_id in {"lean.check", "lean.proof_edit.validate"}:
        lean = LeanService(
            store,
            binder.artifacts,
            verification,
            installations,
        )
        resources.append(lean)
        if operation_id == "lean.check":
            return SelectedOperationBinding(
                LeanCheckAdapter(lean, runtime), tuple(resources)
            )
        proof_edit_adapter, _ = install_lean_proof_edit_operation(
            store,
            schemas,
            binder.artifacts,
            lean,
            runtime,
        )
        return SelectedOperationBinding(proof_edit_adapter, tuple(resources))
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


def install_selected_lean_catalog(
    context: CatalogBuildContext,
    *,
    polytope: object | None = None,
    resources: object | None = None,
) -> None:
    """Compile Lean statements, checkers, and exploration operations."""

    del polytope
    _install_lean_statements(context)
    if not (
        context.authorize_bundled_checkers
        or context.checkers.bind_existing_when_omitted
    ):
        return
    if resources is None:
        raise TypeError("lean catalog install requires catalog build resources")
    _install_authorized_lean_operations(
        context, cast("CatalogBuildResources", resources)
    )


def _install_lean_statements(context: CatalogBuildContext) -> None:
    from jacobian.contracts.operations import ProviderAvailability

    lean_runtime = lean_frontend_provider_runtime()
    if lean_runtime.availability is not ProviderAvailability.AVAILABLE:
        return
    lean_adapters, _ = install_lean_statement_operations(
        context.store,
        context.schemas,
        context.artifacts,
        provider_runtime=lean_runtime,
    )
    for lean_adapter in lean_adapters:
        context.register_operation(lean_adapter)


def _install_authorized_lean_operations(
    context: CatalogBuildContext,
    catalog_resources: CatalogBuildResources,
) -> None:
    from jacobian.checker_authorization import install_lean_checkers
    from jacobian.contracts.operations import ProviderAvailability
    from jacobian.provider_runtime import jacobian_provider_runtime

    lean_checkers, checker_runtime = install_lean_checkers(
        context.store,
        context.schemas,
        context.checkers,
        resolve_provider_runtime=lambda profiles: lean_provider_runtime(
            profiles=profiles,
            checker_ids=(),
        ),
    )
    runtime = checker_runtime.model_copy(
        update={
            "checker_ids": tuple(
                installation.checker_id
                for _, installation in sorted(
                    lean_checkers.items(),
                    key=lambda item: item[0].value,
                )
                if installation.checker_id is not None
            )
        }
    )
    context.register_operation(
        install_lean_proof_state_inspect_only(
            context.store,
            context.schemas,
            context.artifacts,
            lean_checkers,
            jacobian_provider_runtime(
                "jacobian.lean4",
                features=("immutable-proof-state", "read-only-inspection"),
            ),
        )
    )
    if runtime.availability is not ProviderAvailability.AVAILABLE:
        _LOGGER.warning("lean.check is not installed: %s", runtime.diagnostic)
        return
    if any(installation.checker_id is None for installation in lean_checkers.values()):
        _LOGGER.warning("lean.check is not installed: no active Lean checker")
        return
    try:
        catalog_resources.lean_declarations = installed_lean_declaration_service(
            runtime,
            cache_root=context.store.root / "cache" / "lean-declarations",
        )
    except (OSError, RuntimeError) as exc:
        _LOGGER.warning("Lean declaration discovery is not installed: %s", exc)
    if catalog_resources.lean_declarations is not None:
        bound_queries = context.binder.bind(
            lean_declaration_query_operations(catalog_resources.lean_declarations)
        )
        for adapter in bound_queries.adapters:
            context.register_operation(adapter)
    catalog_resources.lean = LeanService(
        context.store,
        context.artifacts,
        context.verification,
        lean_checkers,
    )
    context.register_operation(LeanCheckAdapter(catalog_resources.lean, runtime))
    proof_axioms_adapter, _ = install_lean_proof_axioms_operation(
        context.store,
        context.schemas,
        context.artifacts,
        lean_checkers,
        runtime,
    )
    context.register_operation(proof_axioms_adapter)
    adapters, catalog_resources.lean_exploration = install_lean_exploration_operations(
        context.store,
        context.schemas,
        context.artifacts,
        lean_checkers,
        runtime,
    )
    for adapter in adapters:
        if adapter.descriptor.operation_id == "lean.proof_state.inspect":
            continue
        context.register_operation(adapter)
    proof_edit_adapter, _ = install_lean_proof_edit_operation(
        context.store,
        context.schemas,
        context.artifacts,
        catalog_resources.lean,
        runtime,
    )
    context.register_operation(proof_edit_adapter)


__all__ = [
    "SELECTED_LEAN_OPERATION_IDS",
    "bind_selected_lean_operation",
    "install_selected_lean_catalog",
]
