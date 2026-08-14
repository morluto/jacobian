"""Explicit initialization and update lifecycle for deployment-owned state."""

from __future__ import annotations

import sqlite3
from enum import StrEnum
from pathlib import Path

from jacobian import __version__
from jacobian.contracts.operations import OperationDescriptor
from jacobian.operation_catalog import (
    CatalogBuildResult,
    CompiledCatalogEntry,
    OperationCatalog,
    OperationCatalogError,
    OperationCatalogStore,
    declaration_digest,
    exact_checker_declaration_digest,
    operation_declaration_digest,
)
from jacobian.operation_service import OperationPolicy
from jacobian.persistence.migrations import (
    CURRENT_STATE_FORMAT_REVISION,
    STATE_MIGRATIONS,
    SUPPORTED_STATE_FLOOR,
)
from jacobian.persistence.state_health import StateHealth, inspect_state_health
from jacobian.portfolio.builtin import load_builtin_operation_modules
from jacobian.registry import CheckerRegistry
from jacobian.runtime import CheckerAuthorityMode, create_runtime


class CheckerAuthorization(StrEnum):
    BUNDLED = "bundled"
    NONE = "none"


def initialize_state(
    state_dir: Path,
    *,
    checker_authorization: CheckerAuthorization = CheckerAuthorization.BUNDLED,
) -> CatalogBuildResult:
    """Create current state, or return its already-current catalog summary."""

    health = _health(state_dir)
    if health.status not in {"MISSING", "UNINITIALIZED", "COMPATIBLE"}:
        raise OperationCatalogError(
            "STATE_UPDATE_REQUIRED: existing state requires `jacobian update`"
        )
    if health.status == "COMPATIBLE":
        current = _load_current_catalog(state_dir)
        if current is not None:
            return CatalogBuildResult(
                revision=current.header.revision,
                operation_count=len(current.snapshot().operations),
                omitted_operations=(),
                diagnostics=current.header.diagnostics,
            )
    return _build_catalog(state_dir, checker_authorization)


def update_state(
    state_dir: Path,
    *,
    checker_authorization: CheckerAuthorization = CheckerAuthorization.BUNDLED,
) -> CatalogBuildResult:
    """Migrate existing state, reauthorize as requested, and select a new catalog."""

    health = _health(state_dir)
    if health.status in {"MISSING", "UNINITIALIZED"}:
        raise OperationCatalogError(
            "STATE_INITIALIZATION_REQUIRED: state does not exist; run `jacobian init`"
        )
    if health.status in {"INCOMPATIBLE", "UNSUPPORTED", "CORRUPT", "UNREADABLE"}:
        raise OperationCatalogError(
            "STATE_UPDATE_REQUIRED: state cannot be updated safely: "
            + (health.diagnostic or health.status)
        )
    return _build_catalog(state_dir, checker_authorization)


def _health(state_dir: Path) -> StateHealth:
    return inspect_state_health(
        state_dir,
        STATE_MIGRATIONS,
        supported_floor=SUPPORTED_STATE_FLOOR,
        current_revision=CURRENT_STATE_FORMAT_REVISION,
    )


def _load_current_catalog(state_dir: Path) -> OperationCatalog | None:
    try:
        return OperationCatalog(
            state_dir / "metadata.sqlite3",
            OperationPolicy(),
            expected_package_version=__version__,
        )
    except OperationCatalogError:
        return None


def _build_catalog(
    state_dir: Path,
    checker_authorization: CheckerAuthorization,
) -> CatalogBuildResult:
    authority = (
        CheckerAuthorityMode.INSTALL_BUNDLED
        if checker_authorization is CheckerAuthorization.BUNDLED
        else CheckerAuthorityMode.NONE
    )
    runtime = create_runtime(state_dir, checker_authority=authority)
    try:
        snapshot = runtime.core.operations.catalog()
        descriptors = tuple(
            sorted(snapshot.operations, key=lambda item: item.operation_id)
        )
        provider_inventory = tuple(
            descriptor.provider_runtime.model_dump(mode="json")
            for descriptor in descriptors
            if descriptor.provider_runtime is not None
        )
        checker_bindings = _checker_bindings(runtime.core.checkers, descriptors)
        loaded_modules = load_builtin_operation_modules()
        declarations = {
            operation.operation_id: (module_name, operation)
            for module_name, operations, _checker_declarations in loaded_modules
            for operation in operations
        }
        exact_verifiers = {
            declaration.verification_operation_id: (module_name, declaration)
            for module_name, _operations, checker_declarations in loaded_modules
            for declaration in checker_declarations
            if declaration.verification_operation_id is not None
        }
        entries = tuple(
            CompiledCatalogEntry(
                descriptor=descriptor,
                declaration_module=(
                    declarations[descriptor.operation_id][0]
                    if descriptor.operation_id in declarations
                    else exact_verifiers[descriptor.operation_id][0]
                    if descriptor.operation_id in exact_verifiers
                    else type(
                        runtime.core.operations._adapters[descriptor.operation_id]
                    ).__module__
                ),
                declaration_digest=(
                    operation_declaration_digest(
                        declarations[descriptor.operation_id][1]
                    )
                    if descriptor.operation_id in declarations
                    else exact_checker_declaration_digest(
                        exact_verifiers[descriptor.operation_id][1],
                        descriptor,
                    )
                    if descriptor.operation_id in exact_verifiers
                    else declaration_digest(
                        {
                            "operation_id": descriptor.operation_id,
                            "version": descriptor.version,
                            "input_schema": descriptor.input_schema,
                            "output_schema": descriptor.output_schema,
                        }
                    )
                ),
            )
            for descriptor in descriptors
        )
        return OperationCatalogStore(state_dir / "metadata.sqlite3").commit(
            package_version=__version__,
            provider_inventory_digest=declaration_digest(
                {"providers": list(provider_inventory)}
            ),
            checker_binding_digest=declaration_digest(
                {
                    "bindings": [
                        [operation_id, [checker_id, digest]]
                        for operation_id, (checker_id, digest) in sorted(
                            checker_bindings.items()
                        )
                    ]
                }
            ),
            entries=entries,
            checker_bindings=checker_bindings,
        )
    finally:
        runtime.close()


def _checker_bindings(
    checkers: CheckerRegistry,
    descriptors: tuple[OperationDescriptor, ...],
) -> dict[str, tuple[str, str]]:
    bindings: dict[str, tuple[str, str]] = {}
    for descriptor in descriptors:
        runtime = descriptor.provider_runtime
        if runtime is None or not runtime.checker_ids:
            continue
        checker_id = str(runtime.checker_ids[0])
        registration = checkers.require_active(checker_id)
        bindings[descriptor.operation_id] = (
            checker_id,
            registration.implementation_digest,
        )
    return bindings


def active_catalog_revision(state_dir: Path) -> int | None:
    """Read the selected revision without constructing execution services."""

    database_path = state_dir / "metadata.sqlite3"
    if not database_path.is_file():
        return None
    try:
        with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT snapshot_revision FROM active_operation_catalog WHERE id = 0"
            ).fetchone()
    except sqlite3.DatabaseError:
        return None
    return None if row is None else int(row[0])


__all__ = [
    "CheckerAuthorization",
    "active_catalog_revision",
    "initialize_state",
    "update_state",
]
