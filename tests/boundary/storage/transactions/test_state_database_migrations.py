"""Upgrade and validation coverage for the SQLite-native migration ledger."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from jacobian.persistence.database import (
    Migration,
    StateDatabase,
    StateDatabaseError,
)
from jacobian.persistence.migrations import STATE_MIGRATIONS
from jacobian.store import ArtifactStore, StoreError


def _ledger(root: Path) -> tuple[sqlite3.Row, ...]:
    connection = sqlite3.connect(root / "metadata.sqlite3")
    connection.row_factory = sqlite3.Row
    try:
        return tuple(
            connection.execute(
                """
                SELECT revision, name, checksum, applied_at
                FROM jacobian_schema_migrations
                ORDER BY revision
                """
            )
        )
    finally:
        connection.close()


def test_fresh_store_records_immutable_ordered_migrations(tmp_path: Path) -> None:
    with ArtifactStore(tmp_path):
        pass

    rows = _ledger(tmp_path)
    assert tuple(row["revision"] for row in rows) == tuple(
        range(1, len(STATE_MIGRATIONS) + 1)
    )
    assert tuple(row["name"] for row in rows) == tuple(
        migration.name for migration in STATE_MIGRATIONS
    )
    assert tuple(row["checksum"] for row in rows) == tuple(
        migration.checksum for migration in STATE_MIGRATIONS
    )
    assert all(row["applied_at"] for row in rows)


def test_legacy_schema_without_ledger_is_adopted(tmp_path: Path) -> None:
    with ArtifactStore(tmp_path):
        pass
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute("DROP TABLE jacobian_schema_migrations")
        connection.commit()
    finally:
        connection.close()

    with ArtifactStore(tmp_path):
        pass

    assert len(_ledger(tmp_path)) == len(STATE_MIGRATIONS)


def test_current_head_bootstrap_performs_no_sqlite_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with ArtifactStore(tmp_path):
        pass

    original_connect = sqlite3.connect
    write_actions: list[int] = []
    forbidden = {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
    }

    def observed_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = original_connect(*args, **kwargs)

        def observe(
            action: int,
            _arg1: str | None,
            _arg2: str | None,
            _database: str | None,
            _trigger: str | None,
        ) -> int:
            if action in forbidden:
                write_actions.append(action)
            return sqlite3.SQLITE_OK

        connection.set_authorizer(observe)
        return connection

    monkeypatch.setattr(sqlite3, "connect", observed_connect)
    with ArtifactStore(tmp_path):
        pass

    assert write_actions == []


@pytest.mark.parametrize("field", ["name", "checksum"])
def test_changed_migration_identity_fails_closed(
    tmp_path: Path,
    field: str,
) -> None:
    with ArtifactStore(tmp_path):
        pass
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute(
            f"UPDATE jacobian_schema_migrations SET {field} = ? WHERE revision = 1",
            ("changed",),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(StoreError, match="schema migration failed"):
        ArtifactStore(tmp_path)


def test_newer_revision_fails_closed(tmp_path: Path) -> None:
    with ArtifactStore(tmp_path):
        pass
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute(
            """
            INSERT INTO jacobian_schema_migrations(revision, name, checksum)
            VALUES (99, 'future', 'sha256:future')
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(StoreError, match="UNSUPPORTED_STATE_VERSION") as exc_info:
        ArtifactStore(tmp_path)
    assert exc_info.value.detected_revision == 99
    assert exc_info.value.minimum_revision == 3


def test_missing_tail_revision_is_reapplied(tmp_path: Path) -> None:
    with ArtifactStore(tmp_path):
        pass
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute(
            "DELETE FROM jacobian_schema_migrations WHERE revision = ?",
            (len(STATE_MIGRATIONS),),
        )
        connection.commit()
    finally:
        connection.close()

    with ArtifactStore(tmp_path):
        pass

    assert tuple(row["revision"] for row in _ledger(tmp_path)) == tuple(
        range(1, len(STATE_MIGRATIONS) + 1)
    )


def test_missing_non_tail_revision_fails_closed(tmp_path: Path) -> None:
    with ArtifactStore(tmp_path):
        pass
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute("DELETE FROM jacobian_schema_migrations WHERE revision = 1")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(StoreError, match="schema migration failed"):
        ArtifactStore(tmp_path)


def test_failed_migration_rolls_back_and_can_be_retried(tmp_path: Path) -> None:
    database_path = tmp_path / "state.sqlite3"

    def install_first(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE first_revision (value TEXT)")

    def fail_second(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE uncommitted_revision (value TEXT)")
        raise RuntimeError("injected migration failure")

    failing = (
        Migration(1, "first", "create first table", install_first),
        Migration(2, "second", "create second table", fail_second),
    )
    database = StateDatabase(database_path, synchronous="FULL")
    with pytest.raises(RuntimeError, match="injected migration failure"):
        database.migrate(failing)
    database.close(checkpoint=False)

    connection = sqlite3.connect(database_path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()
    assert "first_revision" not in tables
    assert "uncommitted_revision" not in tables
    assert "jacobian_schema_migrations" not in tables

    def install_second(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE second_revision (value TEXT)")

    corrected = (
        failing[0],
        Migration(2, "second", "create second table", install_second),
    )
    reopened = StateDatabase(database_path, synchronous="FULL")
    reopened.migrate(corrected)
    assert reopened.open_connection_count == 0
    reopened.close(checkpoint=True)

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM jacobian_schema_migrations"
        ).fetchone() == (2,)
    finally:
        connection.close()


def test_invalid_in_memory_migration_order_is_rejected(tmp_path: Path) -> None:
    migration = Migration(2, "second", "noop", lambda _connection: None)
    database = StateDatabase(tmp_path / "state.sqlite3", synchronous="FULL")
    with pytest.raises(StateDatabaseError, match="ordered, consecutive"):
        database.migrate((migration,))
    database.close(checkpoint=False)


def test_close_cannot_race_connection_configuration_and_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3", synchronous="FULL")
    raw_connection_opened = threading.Event()
    release_connect = threading.Event()
    original_connect = sqlite3.connect

    def blocked_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = original_connect(*args, **kwargs)
        raw_connection_opened.set()
        assert release_connect.wait(timeout=2)
        return connection

    monkeypatch.setattr(sqlite3, "connect", blocked_connect)
    connect_errors: list[BaseException] = []

    def connect() -> None:
        try:
            database.connect()
        except BaseException as exc:
            connect_errors.append(exc)

    connector = threading.Thread(target=connect)
    connector.start()
    assert raw_connection_opened.wait(timeout=1)

    closer = threading.Thread(target=lambda: database.close(checkpoint=False))
    closer.start()
    release_connect.set()
    connector.join(timeout=2)
    closer.join(timeout=2)

    assert not connector.is_alive()
    assert not closer.is_alive()
    assert database.open_connection_count == 0
    with pytest.raises(StateDatabaseError, match="closed"):
        database.connect()
    assert connect_errors == []


def test_two_processes_can_race_to_migrate_empty_state(tmp_path: Path) -> None:
    ready = tmp_path / "start"
    script = """
import sys
import time
from pathlib import Path
from jacobian.store import ArtifactStore

root = Path(sys.argv[1])
ready = Path(sys.argv[2])
while not ready.exists():
    time.sleep(0.005)
with ArtifactStore(root):
    pass
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), str(ready)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    ready.touch()

    failures: list[str] = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        if process.returncode != 0:
            failures.append(f"stdout={stdout!r} stderr={stderr!r}")

    assert failures == []
    assert tuple(row["revision"] for row in _ledger(tmp_path)) == tuple(
        range(1, len(STATE_MIGRATIONS) + 1)
    )
