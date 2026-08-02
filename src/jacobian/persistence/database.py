"""Connection and lifecycle ownership for Jacobian's SQLite state."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from jacobian.persistence.locking import PersistenceLock


class StateDatabaseError(RuntimeError):
    """The SQLite state owner could not complete a lifecycle operation."""


@dataclass(frozen=True, slots=True)
class Migration:
    """One immutable, ordered SQLite state transition."""

    revision: int
    name: str
    definition: str
    apply: Callable[[sqlite3.Connection], None]

    @property
    def checksum(self) -> str:
        framed = f"{self.revision}\0{self.name}\0{self.definition}".encode()
        return "sha256:" + hashlib.sha256(framed).hexdigest()


class _ConnectionState(threading.local):
    def __init__(self) -> None:
        self.transaction: sqlite3.Connection | None = None
        self.connection: sqlite3.Connection | None = None
        self.pending_sync_directories: set[Path] = set()


class StateDatabase:
    """Own SQLite connections, PRAGMAs, reuse, checkpointing, and closure."""

    def __init__(
        self,
        path: Path,
        *,
        synchronous: str,
    ) -> None:
        self.path = path
        self.synchronous = synchronous
        self._state = _ConnectionState()
        self._connection_lock = threading.RLock()
        self._open_connections: set[sqlite3.Connection] = set()
        self._lifecycle_lock = PersistenceLock(
            path.with_name(path.name + ".lifecycle.lock")
        )
        self._closed = False

    @property
    def open_connection_count(self) -> int:
        """Return the number of live handles owned across every thread."""

        with self._connection_lock:
            return len(self._open_connections)

    @property
    def transaction_active(self) -> bool:
        return self._state.transaction is not None

    @property
    def transaction_identity(self) -> int | None:
        transaction = self._state.transaction
        return None if transaction is None else id(transaction)

    @property
    def pending_sync_directories(self) -> set[Path]:
        return self._state.pending_sync_directories

    def connect(self) -> sqlite3.Connection:
        """Open and configure one tracked SQLite handle."""

        with self._connection_lock:
            if self._closed:
                raise StateDatabaseError("state database is closed")
            connection = self._open_configured_connection()
            try:
                self._open_connections.add(connection)
            except BaseException:
                connection.close()
                self._open_connections.discard(connection)
                raise
            return connection

    def _open_configured_connection(self) -> sqlite3.Connection:
        """Open a configured handle without registering it as caller-owned."""

        connection = sqlite3.connect(
            self.path,
            timeout=30,
            check_same_thread=False,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA synchronous = {self.synchronous}")
        except BaseException:
            connection.close()
            raise
        return connection

    def close_connection(self, connection: sqlite3.Connection) -> None:
        """Close one tracked handle."""

        try:
            connection.close()
        finally:
            with self._connection_lock:
                self._open_connections.discard(connection)

    def migrate(
        self,
        migrations: tuple[Migration, ...],
    ) -> None:
        """Validate current head or apply missing revisions under one lock."""

        self._validate_migration_plan(migrations)
        with self.transient_connection() as connection:
            if self._at_migration_head(connection, migrations):
                return
            with self._lifecycle_lock.hold():
                if self._at_migration_head(connection, migrations):
                    return
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS jacobian_schema_migrations (
                            revision INTEGER PRIMARY KEY,
                            name TEXT NOT NULL UNIQUE,
                            checksum TEXT NOT NULL,
                            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                        """,
                    )
                    applied = self._applied_migrations(connection, migrations)
                    for migration in migrations[len(applied) :]:
                        migration.apply(connection)
                        connection.execute(
                            """
                            INSERT INTO jacobian_schema_migrations(
                                revision, name, checksum
                            ) VALUES (?, ?, ?)
                            """,
                            (migration.revision, migration.name, migration.checksum),
                        )
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise

    @contextmanager
    def transient_connection(self) -> Iterator[sqlite3.Connection]:
        """Yield one owned handle and close it when the operation finishes."""

        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            self.close_connection(connection)

    @staticmethod
    def _validate_migration_plan(migrations: tuple[Migration, ...]) -> None:
        revisions = tuple(migration.revision for migration in migrations)
        if revisions != tuple(range(1, len(migrations) + 1)):
            raise StateDatabaseError(
                "state migrations must be an ordered, consecutive revision tuple"
            )
        names = tuple(migration.name for migration in migrations)
        if len(names) != len(set(names)):
            raise StateDatabaseError("state migration names must be unique")

    def _at_migration_head(
        self,
        connection: sqlite3.Connection,
        migrations: tuple[Migration, ...],
    ) -> bool:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
        if journal_mode is None or str(journal_mode[0]).casefold() != "wal":
            return False
        table = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'jacobian_schema_migrations'
            """
        ).fetchone()
        if table is None:
            return False
        return len(self._applied_migrations(connection, migrations)) == len(migrations)

    @staticmethod
    def _applied_migrations(
        connection: sqlite3.Connection,
        migrations: tuple[Migration, ...],
    ) -> tuple[sqlite3.Row, ...]:
        rows = tuple(
            connection.execute(
                """
                SELECT revision, name, checksum
                FROM jacobian_schema_migrations
                ORDER BY revision
                """
            ).fetchall()
        )
        if rows and int(rows[-1]["revision"]) > len(migrations):
            raise StateDatabaseError(
                "state database was created by a newer unsupported revision"
            )
        revisions = tuple(int(row["revision"]) for row in rows)
        if revisions != tuple(range(1, len(rows) + 1)):
            raise StateDatabaseError("state migration ledger has missing revisions")
        expected_prefix = migrations[: len(rows)]
        for row, expected in zip(rows, expected_prefix, strict=True):
            if row["name"] != expected.name or row["checksum"] != expected.checksum:
                raise StateDatabaseError(
                    f"state migration {expected.revision} identity or checksum changed"
                )
        return rows

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield this thread's transaction or reusable owned connection."""

        with self._connection_lock:
            if self._closed:
                raise StateDatabaseError("state database is closed")
            connection = self._state.transaction
            in_transaction = connection is not None
            if connection is None:
                connection = self._state.connection
            if connection is None:
                connection = self.connect()
                self._state.connection = connection
        if in_transaction:
            yield connection
            return
        with connection:
            yield connection

    def activate_transaction(self, connection: sqlite3.Connection) -> None:
        if self._state.transaction is not None:
            raise StateDatabaseError(
                "nested state database transactions are unsupported"
            )
        self._state.transaction = connection

    def clear_transaction(self) -> None:
        self._state.transaction = None
        self._state.pending_sync_directories.clear()

    def close(self, *, checkpoint: bool) -> None:
        """Checkpoint when requested, close every handle, and end ownership."""

        with self._connection_lock:
            if self._closed:
                return
            self._closed = True
            checkpoint_error = self._checkpoint_for_close() if checkpoint else None
            close_error: BaseException | None = None
            try:
                self._close_all_connections()
            except BaseException as exc:
                close_error = exc
            self._state.connection = None
        if checkpoint_error is not None:
            raise StateDatabaseError(
                "could not checkpoint state database"
            ) from checkpoint_error
        if close_error is not None:
            raise close_error

    def _checkpoint_for_close(self) -> BaseException | None:
        with self._lifecycle_lock.hold():
            connection: sqlite3.Connection | None = None
            failure: BaseException | None = None
            try:
                connection = self._open_configured_connection()
                result = connection.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                ).fetchone()
                if result is None or result[0] != 0:
                    failure = StateDatabaseError(
                        f"could not checkpoint state database: {result!r}"
                    )
            except BaseException as exc:
                failure = exc
            finally:
                if connection is not None:
                    try:
                        connection.close()
                    except BaseException as exc:
                        if failure is None:
                            failure = exc
            return failure

    def _close_all_connections(self) -> None:
        failures: list[BaseException] = []
        with self._connection_lock:
            for connection in tuple(self._open_connections):
                try:
                    connection.close()
                except BaseException as exc:
                    failures.append(exc)
                finally:
                    self._open_connections.discard(connection)
        if failures:
            raise StateDatabaseError(
                "could not close every state database handle"
            ) from (failures[0])
