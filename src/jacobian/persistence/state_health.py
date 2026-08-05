"""Read-only diagnostics for the persisted SQLite migration ledger."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from jacobian.persistence.database import Migration

StateHealthStatus = Literal[
    "MISSING",
    "UNINITIALIZED",
    "COMPATIBLE",
    "MIGRATION_PENDING",
    "INCOMPATIBLE",
    "UNSUPPORTED",
    "CORRUPT",
    "UNREADABLE",
]


@dataclass(frozen=True, slots=True)
class MigrationMismatch:
    """One persisted migration identity that differs from the current code."""

    revision: int
    name: str
    stored_checksum: str
    current_checksum: str

    def as_dict(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "name": self.name,
            "stored_checksum": self.stored_checksum,
            "current_checksum": self.current_checksum,
        }


@dataclass(frozen=True, slots=True)
class StateHealth:
    """A bounded, non-mutating report about one state directory."""

    status: StateHealthStatus
    state_dir: str
    database_path: str
    persisted_revision: int | None
    supported_floor: int
    current_revision: int
    mismatches: tuple[MigrationMismatch, ...] = ()
    diagnostic: str | None = None

    @property
    def blocking(self) -> bool:
        return self.status in {
            "INCOMPATIBLE",
            "UNSUPPORTED",
            "CORRUPT",
            "UNREADABLE",
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "state_dir": self.state_dir,
            "database_path": self.database_path,
            "persisted_revision": self.persisted_revision,
            "supported_floor": self.supported_floor,
            "current_revision": self.current_revision,
            "mismatches": tuple(item.as_dict() for item in self.mismatches),
            "diagnostic": self.diagnostic,
            "blocking": self.blocking,
        }


def inspect_state_health(
    state_dir: str | Path,
    migrations: tuple[Migration, ...],
    *,
    supported_floor: int,
    current_revision: int,
) -> StateHealth:
    """Inspect migration metadata without creating or modifying any files."""

    resolved_state_dir = Path(state_dir).resolve()
    database_path = resolved_state_dir / "metadata.sqlite3"
    ledger = _read_migration_ledger(
        resolved_state_dir,
        database_path,
        supported_floor=supported_floor,
        current_revision=current_revision,
    )
    if isinstance(ledger, StateHealth):
        return ledger
    rows = ledger

    if not rows:
        return _health(
            "CORRUPT",
            resolved_state_dir,
            database_path,
            supported_floor,
            current_revision,
            diagnostic="the migration ledger is empty",
        )

    try:
        revisions = tuple(int(str(row[0])) for row in rows)
    except (TypeError, ValueError):
        return _health(
            "CORRUPT",
            resolved_state_dir,
            database_path,
            supported_floor,
            current_revision,
            diagnostic="the migration ledger contains a non-integer revision",
        )
    persisted_revision = revisions[-1]
    if persisted_revision > len(migrations):
        return _health(
            "UNSUPPORTED",
            resolved_state_dir,
            database_path,
            supported_floor,
            current_revision,
            persisted_revision=persisted_revision,
            diagnostic=(
                f"state revision {persisted_revision} is newer than the supported "
                f"revision {current_revision}"
            ),
        )
    if revisions != tuple(range(1, len(rows) + 1)):
        return _health(
            "CORRUPT",
            resolved_state_dir,
            database_path,
            supported_floor,
            current_revision,
            persisted_revision=persisted_revision,
            diagnostic="the migration ledger has missing or reordered revisions",
        )

    mismatches = tuple(
        MigrationMismatch(
            revision=expected.revision,
            name=expected.name,
            stored_checksum=str(row[2]),
            current_checksum=expected.checksum,
        )
        for row, expected in zip(rows, migrations, strict=False)
        if row[1] != expected.name or row[2] != expected.checksum
    )
    if mismatches:
        return _health(
            "INCOMPATIBLE",
            resolved_state_dir,
            database_path,
            supported_floor,
            current_revision,
            persisted_revision=persisted_revision,
            mismatches=mismatches,
            diagnostic=(
                "the persisted migration identity differs from the current "
                "Jacobian version"
            ),
        )
    if persisted_revision < supported_floor:
        return _health(
            "UNSUPPORTED",
            resolved_state_dir,
            database_path,
            supported_floor,
            current_revision,
            persisted_revision=persisted_revision,
            diagnostic=(
                f"state revision {persisted_revision} is below the supported "
                f"floor {supported_floor}"
            ),
        )
    if persisted_revision < current_revision:
        return _health(
            "MIGRATION_PENDING",
            resolved_state_dir,
            database_path,
            supported_floor,
            current_revision,
            persisted_revision=persisted_revision,
            diagnostic="the state has unapplied migrations",
        )
    return _health(
        "COMPATIBLE",
        resolved_state_dir,
        database_path,
        supported_floor,
        current_revision,
        persisted_revision=persisted_revision,
    )


def _read_migration_ledger(
    state_dir: Path,
    database_path: Path,
    *,
    supported_floor: int,
    current_revision: int,
) -> tuple[tuple[object, ...], ...] | StateHealth:
    if not database_path.exists():
        return _health(
            "MISSING",
            state_dir,
            database_path,
            supported_floor,
            current_revision,
        )
    if not database_path.is_file():
        return _health(
            "UNREADABLE",
            state_dir,
            database_path,
            supported_floor,
            current_revision,
            diagnostic="metadata.sqlite3 is not a regular file",
        )
    try:
        uri = f"file:{quote(str(database_path), safe='/')}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            table = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'jacobian_schema_migrations'
                """
            ).fetchone()
            if table is None:
                return _health(
                    "UNINITIALIZED",
                    state_dir,
                    database_path,
                    supported_floor,
                    current_revision,
                )
            return tuple(
                tuple(row)
                for row in connection.execute(
                    """
                    SELECT revision, name, checksum
                    FROM jacobian_schema_migrations
                    ORDER BY revision
                    """
                ).fetchall()
            )
    except (OSError, sqlite3.DatabaseError) as exc:
        return _health(
            "UNREADABLE",
            state_dir,
            database_path,
            supported_floor,
            current_revision,
            diagnostic=f"could not read the migration ledger: {exc}",
        )


def _health(
    status: StateHealthStatus,
    state_dir: Path,
    database_path: Path,
    supported_floor: int,
    current_revision: int,
    *,
    persisted_revision: int | None = None,
    mismatches: tuple[MigrationMismatch, ...] = (),
    diagnostic: str | None = None,
) -> StateHealth:
    return StateHealth(
        status=status,
        state_dir=str(state_dir),
        database_path=str(database_path),
        persisted_revision=persisted_revision,
        supported_floor=supported_floor,
        current_revision=current_revision,
        mismatches=mismatches,
        diagnostic=diagnostic,
    )


__all__ = ["MigrationMismatch", "StateHealth", "inspect_state_health"]
