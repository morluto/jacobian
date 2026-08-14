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
from jacobian.persistence.migrations import (
    CURRENT_STATE_FORMAT_REVISION,
    STATE_MIGRATIONS,
    SUPPORTED_STATE_FLOOR,
)
from jacobian.storage.errors import StorageError, UnsupportedStateVersionError
from jacobian.storage.repository import ArtifactRepository


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
    with ArtifactRepository(tmp_path):
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

    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert not any(name.startswith("workspace") for name in tables)
        legacy_episode_prefix = "research_" + "episode"
        assert not any(name.startswith(legacy_episode_prefix) for name in tables)
        assert {
            "operation_catalog_snapshots",
            "operation_catalog_entries",
            "active_operation_catalog",
            "operation_checker_bindings",
        } <= tables
        assert (
            not {
                "experiments",
                "search_experiments",
                "installed_plugins",
                "reasoning_runs",
                "reasoning_events",
            }
            & tables
        )
        assert connection.execute(
            "SELECT format_revision FROM jacobian_state_format WHERE id = 0"
        ).fetchone() == (CURRENT_STATE_FORMAT_REVISION,)
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jacobian_data_upgrades'"
            ).fetchone()
            is None
        )
    finally:
        connection.close()


def test_revision_twelve_preserves_artifacts_and_retires_runtime_tables(
    tmp_path: Path,
) -> None:
    with ArtifactRepository(tmp_path):
        pass

    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute(
            "DELETE FROM jacobian_schema_migrations WHERE revision >= 12"
        )
        connection.execute(
            "UPDATE jacobian_state_format SET format_revision = 11 WHERE id = 0"
        )
        for table in (
            "operation_checker_bindings",
            "active_operation_catalog",
            "operation_catalog_entries",
            "operation_catalog_snapshots",
        ):
            connection.execute(f"DROP TABLE {table}")
        connection.execute(
            """
            INSERT INTO artifacts(
                artifact_uri, manifest_digest, object_digest, payload_digest,
                schema_uri, semantics_uri, canonicalizer_digest, summary
            ) VALUES (
                'artifact://sha256/legacy', 'sha256:manifest', 'sha256:object',
                'sha256:payload', 'schema://legacy', 'semantics://legacy',
                'sha256:canonicalizer', 'legacy artifact'
            )
            """
        )
        connection.commit()
    finally:
        connection.close()

    database = StateDatabase(tmp_path / "metadata.sqlite3", synchronous="FULL")
    try:
        database.migrate(STATE_MIGRATIONS)
    finally:
        database.close(checkpoint=False)

    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        assert connection.execute(
            "SELECT summary FROM artifacts WHERE artifact_uri = ?",
            ("artifact://sha256/legacy",),
        ).fetchone() == ("legacy artifact",)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "reasoning_runs" not in tables
        assert "operation_catalog_snapshots" in tables
        assert connection.execute(
            "SELECT format_revision FROM jacobian_state_format WHERE id = 0"
        ).fetchone() == (CURRENT_STATE_FORMAT_REVISION,)
    finally:
        connection.close()


def test_revision_thirteen_records_overlay_only_catalog_boundary(
    tmp_path: Path,
) -> None:
    with ArtifactRepository(tmp_path):
        pass

    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute("DELETE FROM jacobian_schema_migrations WHERE revision = 13")
        connection.execute(
            "UPDATE jacobian_state_format SET format_revision = 12 WHERE id = 0"
        )
        connection.commit()
    finally:
        connection.close()

    database = StateDatabase(tmp_path / "metadata.sqlite3", synchronous="FULL")
    try:
        database.migrate(STATE_MIGRATIONS)
    finally:
        database.close(checkpoint=False)

    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        assert connection.execute(
            "SELECT format_revision FROM jacobian_state_format WHERE id = 0"
        ).fetchone() == (CURRENT_STATE_FORMAT_REVISION,)
        assert connection.execute(
            "SELECT revision FROM jacobian_schema_migrations WHERE revision = 13"
        ).fetchone() == (13,)
    finally:
        connection.close()


def test_revision_five_removes_populated_legacy_workspace_tables(
    tmp_path: Path,
) -> None:
    with ArtifactRepository(tmp_path):
        pass

    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("DELETE FROM jacobian_schema_migrations WHERE revision > 4")
        connection.execute(
            "UPDATE jacobian_state_format SET format_revision = 4 WHERE id = 0"
        )
        STATE_MIGRATIONS[2].apply(connection)
        connection.commit()

        connection.execute(
            """
            INSERT INTO artifacts(
                artifact_uri,
                manifest_digest,
                object_digest,
                payload_digest,
                schema_uri,
                semantics_uri,
                canonicalizer_digest,
                summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "artifact://legacy-workspace-revision",
                "sha256:legacy-workspace-manifest",
                "sha256:legacy-workspace-object",
                "sha256:legacy-workspace-payload",
                "schema://legacy-workspace",
                "semantics://legacy-workspace",
                "sha256:legacy-workspace-canonicalizer",
                "legacy workspace revision",
            ),
        )
        connection.commit()

        # These tables contain a deliberately cyclic, but valid, legacy row
        # set. The migration must remove it without relying on drop order.
        connection.execute("PRAGMA defer_foreign_keys = ON")
        connection.execute("BEGIN")
        connection.execute(
            """
            INSERT INTO workspaces(workspace_id, name, root_branch_id, created_at)
            VALUES ('workspace-1', 'Legacy workspace', 'branch-1', '2026-08-03')
            """
        )
        connection.execute(
            """
            INSERT INTO workspace_branches(
                branch_id, workspace_id, alias, head_revision_id, created_at
            ) VALUES ('branch-1', 'workspace-1', 'main', 'revision-1', '2026-08-03')
            """
        )
        connection.execute(
            """
            INSERT INTO workspace_revisions(
                revision_id,
                revision_artifact_uri,
                workspace_id,
                branch_id,
                parent_revision_id,
                request_digest,
                created_at
            ) VALUES (
                'revision-1',
                'artifact://legacy-workspace-revision',
                'workspace-1',
                'branch-1',
                NULL,
                'sha256:legacy-workspace-request',
                '2026-08-03'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO workspace_idempotency(
                idempotency_key, operation, request_digest, response_json
            ) VALUES ('idempotency-1', 'create', 'sha256:request', '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO workspace_scratch(
                scratch_id,
                workspace_id,
                branch_id,
                created_revision_id,
                payload_json,
                created_at
            ) VALUES (
                'scratch-1', 'workspace-1', 'branch-1', 'revision-1', '{}',
                '2026-08-03'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO workspace_findings(
                card_id,
                workspace_id,
                branch_id,
                kind,
                created_revision_id,
                payload_json,
                created_at
            ) VALUES (
                'finding-1', 'workspace-1', 'branch-1', 'note', 'revision-1',
                '{}', '2026-08-03'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO workspace_attempts(
                attempt_id,
                workspace_id,
                branch_id,
                target_card_id,
                outcome,
                created_revision_id,
                payload_json,
                created_at
            ) VALUES (
                'attempt-1', 'workspace-1', 'branch-1', 'finding-1', 'open',
                'revision-1', '{}', '2026-08-03'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO workspace_marks(
                mark_id,
                workspace_id,
                branch_id,
                target_card_id,
                state,
                created_revision_id,
                payload_json,
                created_at
            ) VALUES (
                'mark-1', 'workspace-1', 'branch-1', 'finding-1', 'open',
                'revision-1', '{}', '2026-08-03'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO workspace_focus(
                branch_id, workspace_id, updated_revision_id, payload_json
            ) VALUES ('branch-1', 'workspace-1', 'revision-1', '{}')
            """
        )
        connection.commit()
    finally:
        connection.close()

    database = StateDatabase(tmp_path / "metadata.sqlite3", synchronous="FULL")
    try:
        database.migrate(STATE_MIGRATIONS)
    finally:
        database.close(checkpoint=False)

    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert not any(name.startswith("workspace") for name in tables)
        assert connection.execute(
            "SELECT format_revision FROM jacobian_state_format WHERE id = 0"
        ).fetchone() == (CURRENT_STATE_FORMAT_REVISION,)
        assert connection.execute(
            "SELECT revision FROM jacobian_schema_migrations WHERE revision = 5"
        ).fetchone() == (5,)
        assert connection.execute(
            "SELECT revision FROM jacobian_schema_migrations WHERE revision = 6"
        ).fetchone() == (6,)
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jacobian_data_upgrades'"
            ).fetchone()
            is None
        )
    finally:
        connection.close()


def test_legacy_schema_without_ledger_is_adopted(tmp_path: Path) -> None:
    with ArtifactRepository(tmp_path):
        pass
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute("DROP TABLE jacobian_schema_migrations")
        connection.commit()
    finally:
        connection.close()

    with ArtifactRepository(tmp_path):
        pass

    assert len(_ledger(tmp_path)) == len(STATE_MIGRATIONS)


def test_revision_five_state_requires_export_to_current_floor(tmp_path: Path) -> None:
    with ArtifactRepository(tmp_path):
        pass
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute("DELETE FROM jacobian_schema_migrations WHERE revision > 5")
        connection.execute(
            "UPDATE jacobian_state_format SET format_revision = 5 WHERE id = 0"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(UnsupportedStateVersionError) as exc_info:
        ArtifactRepository(tmp_path)
    assert exc_info.value.detected_revision == 5
    assert exc_info.value.minimum_revision == SUPPORTED_STATE_FLOOR


def test_previous_pre_stable_head_requires_fresh_store(tmp_path: Path) -> None:
    with ArtifactRepository(tmp_path):
        pass
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute("DELETE FROM jacobian_schema_migrations WHERE revision = 8")
        connection.execute(
            "UPDATE jacobian_state_format SET format_revision = 7 WHERE id = 0"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(UnsupportedStateVersionError) as exc_info:
        ArtifactRepository(tmp_path)
    assert exc_info.value.detected_revision == 7
    assert exc_info.value.minimum_revision == SUPPORTED_STATE_FLOOR


def test_record_v3_state_requires_its_matching_checkout(tmp_path: Path) -> None:
    with ArtifactRepository(tmp_path):
        pass
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute("DELETE FROM jacobian_schema_migrations WHERE revision = 11")
        connection.execute(
            "UPDATE jacobian_state_format SET format_revision = 10 WHERE id = 0"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(UnsupportedStateVersionError) as exc_info:
        ArtifactRepository(tmp_path)
    assert exc_info.value.detected_revision == 10
    assert exc_info.value.minimum_revision == SUPPORTED_STATE_FLOOR


def test_revision_eleven_requires_matching_checkout(tmp_path: Path) -> None:
    with ArtifactRepository(tmp_path):
        pass
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute(
            "DELETE FROM jacobian_schema_migrations WHERE revision >= 12"
        )
        connection.execute(
            "UPDATE jacobian_state_format SET format_revision = 11 WHERE id = 0"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(UnsupportedStateVersionError) as exc_info:
        ArtifactRepository(tmp_path)
    assert exc_info.value.detected_revision == 11
    assert exc_info.value.minimum_revision == SUPPORTED_STATE_FLOOR


def test_checker_distribution_identity_rejects_existing_authorizations(
    tmp_path: Path,
) -> None:
    with ArtifactRepository(tmp_path):
        pass
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute(
            "DELETE FROM jacobian_schema_migrations WHERE revision >= 10"
        )
        connection.execute(
            "UPDATE jacobian_state_format SET format_revision = 9 WHERE id = 0"
        )
        connection.execute(
            """
            INSERT INTO checkers(
                checker_id, registration_json, authorized, implementation_digest
            ) VALUES (?, ?, ?, ?)
            """,
            (
                "checker://sha256/legacy-authorization",
                b"{}",
                1,
                "sha256:legacy-implementation",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    database = StateDatabase(tmp_path / "metadata.sqlite3", synchronous="FULL")
    try:
        with pytest.raises(
            RuntimeError,
            match="checker distribution identity requires a fresh state directory",
        ):
            database.migrate(STATE_MIGRATIONS)
    finally:
        database.close(checkpoint=False)


def test_current_head_bootstrap_performs_no_sqlite_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with ArtifactRepository(tmp_path):
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
    with ArtifactRepository(tmp_path):
        pass

    assert write_actions == []


@pytest.mark.parametrize("field", ["name", "checksum"])
def test_changed_migration_identity_fails_closed(
    tmp_path: Path,
    field: str,
) -> None:
    with ArtifactRepository(tmp_path):
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

    with pytest.raises(StorageError, match="schema migration failed"):
        ArtifactRepository(tmp_path)


def test_newer_revision_fails_closed(tmp_path: Path) -> None:
    with ArtifactRepository(tmp_path):
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

    with pytest.raises(StorageError, match="UNSUPPORTED_STATE_VERSION") as exc_info:
        ArtifactRepository(tmp_path)
    assert exc_info.value.detected_revision == 99
    assert exc_info.value.minimum_revision == SUPPORTED_STATE_FLOOR


def test_missing_retirement_revision_requires_fresh_store(tmp_path: Path) -> None:
    with ArtifactRepository(tmp_path):
        pass
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute(
            "DELETE FROM jacobian_schema_migrations WHERE revision >= 6",
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(UnsupportedStateVersionError) as exc_info:
        ArtifactRepository(tmp_path)
    assert exc_info.value.detected_revision == 5
    assert exc_info.value.minimum_revision == SUPPORTED_STATE_FLOOR


def test_missing_non_tail_revision_fails_closed(tmp_path: Path) -> None:
    with ArtifactRepository(tmp_path):
        pass
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute("DELETE FROM jacobian_schema_migrations WHERE revision = 1")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(StorageError, match="schema migration failed"):
        ArtifactRepository(tmp_path)


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


def test_close_preserves_handle_cleanup_failure_after_checkpoint_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3", synchronous="FULL")
    checkpoint_failure = sqlite3.OperationalError("injected checkpoint failure")
    close_failure = StateDatabaseError("injected handle close failure")

    monkeypatch.setattr(
        database,
        "_checkpoint_for_close",
        lambda: checkpoint_failure,
    )

    def fail_close_all_connections() -> None:
        raise close_failure

    monkeypatch.setattr(database, "_close_all_connections", fail_close_all_connections)

    with pytest.raises(StateDatabaseError, match="could not checkpoint") as exc:
        database.close(checkpoint=True)

    assert exc.value.__cause__ is checkpoint_failure
    assert exc.value.__notes__ == [
        "state database handle cleanup also failed: injected handle close failure"
    ]


def test_two_processes_can_race_to_migrate_empty_state(tmp_path: Path) -> None:
    ready = tmp_path / "start"
    script = """
import sys
import time
from pathlib import Path
from jacobian.storage.repository import ArtifactRepository

root = Path(sys.argv[1])
ready = Path(sys.argv[2])
while not ready.exists():
    time.sleep(0.005)
with ArtifactRepository(root):
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
