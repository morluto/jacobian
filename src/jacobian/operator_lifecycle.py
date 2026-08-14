"""Explicit initialization and update lifecycle for deployment-owned state."""

from __future__ import annotations

import sqlite3
from enum import StrEnum
from pathlib import Path

from jacobian import __version__
from jacobian.operation_catalog import (
    CatalogBuildResult,
    OperationCatalog,
    OperationCatalogError,
    omitted_packaged_operations,
)
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.persistence.migrations import (
    CURRENT_STATE_FORMAT_REVISION,
    STATE_MIGRATIONS,
    SUPPORTED_STATE_FLOOR,
)
from jacobian.persistence.state_health import StateHealth, inspect_state_health
from jacobian.serving_catalog import ServingCatalog


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
            serving = _load_serving_catalog(state_dir)
            return CatalogBuildResult(
                revision=current.header.revision,
                operation_count=len(serving.snapshot().operations),
                omitted_operations=tuple(
                    sorted(omitted_packaged_operations(current.header.diagnostics))
                ),
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
            OperationVisibilityPolicy(),
            expected_package_version=__version__,
        )
    except OperationCatalogError:
        return None


def _load_serving_catalog(state_dir: Path) -> ServingCatalog:
    try:
        return ServingCatalog.open(
            state_dir / "metadata.sqlite3",
            OperationVisibilityPolicy(),
            expected_package_version=__version__,
        )
    except OperationCatalogError as exc:
        if "overlay is stale" in str(exc):
            raise OperationCatalogError(
                "STATE_UPDATE_REQUIRED: operation catalog overlay is stale; "
                "run `jacobian update`"
            ) from exc
        raise


def _build_catalog(
    state_dir: Path,
    checker_authorization: CheckerAuthorization,
) -> CatalogBuildResult:
    # Compile-only import: operator command import must not assemble the catalog.
    from jacobian.catalog.compiler import compile_operation_catalog

    return compile_operation_catalog(
        state_dir,
        authorize_bundled_checkers=(
            checker_authorization is CheckerAuthorization.BUNDLED
        ),
    )


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
