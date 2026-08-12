"""Transaction coordination and durable recovery markers for artifact storage."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from jacobian.persistence import PersistenceLock, StateDatabase, StateDatabaseError
from jacobian.storage.errors import StorageClosedError, StorageError

if TYPE_CHECKING:
    from jacobian.storage.blobs import FilesystemBlobStore

_LOGGER = logging.getLogger(__name__)


class _ActiveTransactionPaths(threading.local):
    """Database paths transaction-owned by the current thread."""

    def __init__(self) -> None:
        self.paths: set[Path] = set()


_ACTIVE_TRANSACTION_PATHS = _ActiveTransactionPaths()


def transaction_active_for(database_path: str | Path) -> bool:
    """Whether this thread owns a storage transaction for one database."""

    return Path(database_path).resolve() in _ACTIVE_TRANSACTION_PATHS.paths


class ArtifactTransactions:
    """Coordinate SQLite transactions with the concrete filesystem CAS."""

    def __init__(
        self,
        *,
        blob_root: Path,
        db_path: Path,
        transaction_recovery_path: Path,
        database: StateDatabase,
        blob_lock: PersistenceLock,
    ) -> None:
        self.blob_root = blob_root
        self.db_path = db_path
        self.transaction_recovery_path = transaction_recovery_path
        self.database = database
        self._blob_lock = blob_lock
        self._closed = False
        self._recovery_required = False

    def close(self) -> None:
        """Checkpoint SQLite and end this store's owned lifetime."""

        if self._closed:
            return
        if self.transaction_active:
            raise StorageError("cannot close an artifact store during a transaction")
        try:
            self.database.close(checkpoint=not self._recovery_required)
        except StateDatabaseError as exc:
            raise StorageError("could not close artifact store database") from exc
        self._closed = True

    def ensure_open(self) -> None:
        if self._closed:
            raise StorageClosedError("artifact store is closed")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield this thread's transaction connection or one owned connection."""

        if self._closed:
            raise StorageClosedError("artifact store is closed")
        if self._recovery_required:
            raise StorageError(
                "artifact store requires recovery by a fresh ArtifactRepository instance"
            )
        with self.database.connection() as connection:
            yield connection

    @property
    def transaction_active(self) -> bool:
        """Whether this thread is inside an explicit store transaction."""

        return self.database.transaction_active

    @property
    def transaction_identity(self) -> int | None:
        """Process-local identity of this thread's active transaction."""

        return self.database.transaction_identity

    @contextmanager
    def exclusive_blob_lock(self) -> Iterator[None]:
        """Serialize quota accounting and blob publication across processes."""

        with self._blob_lock.hold():
            yield

    def _rollback_transaction(
        self,
        connection: sqlite3.Connection,
        blobs: FilesystemBlobStore,
    ) -> None:
        """Roll back, close the connection, and reconcile blob quota on failure."""

        cleanup_error: BaseException | None = None
        try:
            connection.rollback()
        except BaseException as exc:
            cleanup_error = exc
        if cleanup_error is None:
            try:
                self._flush_transaction_directories()
            except BaseException as exc:
                cleanup_error = exc
        try:
            self.database.close_connection(connection)
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
        finally:
            self._clear_transaction_state()
        if cleanup_error is not None:
            self._recovery_required = True
            raise StorageError(
                "artifact transaction cleanup was not durable; "
                "reopen the store to recover"
            ) from cleanup_error
        try:
            blobs.reconcile_quota(force=True)
        except BaseException:
            self._recovery_required = True
            raise

    def _commit_transaction(self, connection: sqlite3.Connection) -> None:
        """Flush directories, commit, and remove the recovery marker."""

        try:
            self._flush_transaction_directories()
            connection.commit()
            self.database.close_connection(connection)
        except BaseException as exc:
            self._recovery_required = True
            try:
                self.database.close_connection(connection)
            except BaseException:
                _LOGGER.exception("failed to close an uncertain artifact transaction")
            raise StorageError(
                "artifact transaction commit was not durable; "
                "reopen the store to recover"
            ) from exc
        finally:
            self._clear_transaction_state()
        try:
            self.remove_recovery_marker()
        except BaseException:
            self._recovery_required = True
            raise

    def _handle_transaction_failure(
        self,
        connection: sqlite3.Connection | None,
    ) -> None:
        """Clean up a connection after a transaction setup or body failure."""

        if connection is None:
            return
        if self.database.transaction_active:
            try:
                self.database.close_connection(connection)
            except BaseException:
                _LOGGER.exception(
                    "failed to close artifact transaction during setup cleanup"
                )
            self._clear_transaction_state()
        elif self.transaction_recovery_path.exists() and not self._recovery_required:
            try:
                self.database.close_connection(connection)
            except BaseException:
                _LOGGER.exception(
                    "failed to close artifact transaction after setup failure"
                )
            self._recovery_required = True

    @contextmanager
    def transaction(self, blobs: FilesystemBlobStore) -> Iterator[None]:
        """Commit related store operations through one SQLite transaction.

        Blob publication remains content-addressed and durable. If metadata
        rolls back, quota accounting is reconciled against the published blob
        set before control returns to the caller.
        """

        if self._closed:
            raise StorageClosedError("artifact store is closed")
        if self._recovery_required:
            raise StorageError(
                "artifact store requires recovery by a fresh ArtifactRepository instance"
            )
        if self.database.transaction_active:
            raise StorageError("nested artifact store transactions are unsupported")

        with self.exclusive_blob_lock():
            connection: sqlite3.Connection | None = None
            try:
                if self.transaction_recovery_path.exists():
                    blobs.reconcile_quota(force=True)
                self.write_recovery_marker()
                connection = self.database.connect()
            except BaseException:
                self._recovery_required = True
                raise
            try:
                connection.execute("BEGIN IMMEDIATE")
                self.database.activate_transaction(connection)
                _ACTIVE_TRANSACTION_PATHS.paths.add(self.db_path)
                try:
                    yield
                except BaseException:
                    self._rollback_transaction(connection, blobs)
                    raise
                else:
                    self._commit_transaction(connection)
            except BaseException:
                self._handle_transaction_failure(connection)
                raise

    def _clear_transaction_state(self) -> None:
        """Release process-local ownership even when durable cleanup fails."""

        self.database.clear_transaction()
        _ACTIVE_TRANSACTION_PATHS.paths.discard(self.db_path)

    def sync_directory(self, path: Path) -> None:
        if os.name == "nt":  # pragma: no cover - Windows has no directory fsync
            return
        descriptor = -1
        try:
            descriptor = os.open(path, os.O_RDONLY)
            os.fsync(descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def sync_blob_publication_directories(
        self,
        prefix: Path,
        *,
        prefix_created: bool,
    ) -> None:
        directories = self.database.pending_sync_directories
        if self.transaction_active:
            directories.add(prefix)
            if prefix_created:
                directories.add(self.blob_root)
            return

        self.sync_directory(prefix)
        if prefix_created:
            self.sync_directory(self.blob_root)

    def _flush_transaction_directories(self) -> None:
        """Make published blob names durable before committing their metadata."""

        directories = self.database.pending_sync_directories
        for directory in sorted(
            directories,
            key=lambda path: (len(path.parts), str(path)),
            reverse=True,
        ):
            self.sync_directory(directory)
        directories.clear()

    def write_recovery_marker(self) -> None:
        with self.transaction_recovery_path.open("wb") as marker:
            marker.write(b"jacobian artifact transaction in progress\n")
            marker.flush()
            os.fsync(marker.fileno())
        self.sync_directory(self.transaction_recovery_path.parent)

    def remove_recovery_marker(self) -> None:
        self.transaction_recovery_path.unlink(missing_ok=True)
        self.sync_directory(self.transaction_recovery_path.parent)
