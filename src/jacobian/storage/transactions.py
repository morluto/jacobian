"""Transaction coordination and durable recovery markers for artifact storage."""

# The concrete owner is ArtifactRepository; this component owns transaction
# protocol while the composition root supplies database and blob collaborators.
# mypy: ignore_errors = True

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Self

from jacobian.persistence import StateDatabaseError
from jacobian.storage.errors import StorageClosedError, StorageError

_LOGGER = logging.getLogger(__name__)


class _ActiveTransactionPaths(threading.local):
    """Database paths transaction-owned by the current thread."""

    def __init__(self) -> None:
        self.paths: set[Path] = set()


_ACTIVE_TRANSACTION_PATHS = _ActiveTransactionPaths()


def transaction_active_for(database_path: str | Path) -> bool:
    """Whether this thread owns a storage transaction for one database."""

    return Path(database_path).resolve() in _ACTIVE_TRANSACTION_PATHS.paths


class TransactionCoordinator:
    """Own transaction sequencing, directory sync, and recovery markers."""

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

    def __enter__(self) -> Self:
        if self._closed:
            raise StorageClosedError("artifact store is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

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

    def _rollback_transaction(self, connection: sqlite3.Connection) -> None:
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
            self._reconcile_blob_quota(force=True)
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
            self._remove_transaction_recovery_marker()
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
    def transaction(self) -> Iterator[None]:
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

        with self._exclusive_blob_lock():
            connection: sqlite3.Connection | None = None
            try:
                if self.transaction_recovery_path.exists():
                    self._reconcile_blob_quota(force=True)
                self._write_transaction_recovery_marker()
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
                    self._rollback_transaction(connection)
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

    def _sync_directory(self, path: Path) -> None:
        if os.name == "nt":  # pragma: no cover - Windows has no directory fsync
            return
        descriptor = -1
        try:
            descriptor = os.open(path, os.O_RDONLY)
            os.fsync(descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _sync_root_directory(self) -> None:
        self._sync_directory(self.root)

    def _sync_blob_publication_directories(
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

        self._sync_directory(prefix)
        if prefix_created:
            self._sync_directory(self.blob_root)

    def _flush_transaction_directories(self) -> None:
        """Make published blob names durable before committing their metadata."""

        directories = self.database.pending_sync_directories
        for directory in sorted(
            directories,
            key=lambda path: (len(path.parts), str(path)),
            reverse=True,
        ):
            self._sync_directory(directory)
        directories.clear()

    def _write_transaction_recovery_marker(self) -> None:
        with self.transaction_recovery_path.open("wb") as marker:
            marker.write(b"jacobian artifact transaction in progress\n")
            marker.flush()
            os.fsync(marker.fileno())
        self._sync_root_directory()

    def _remove_transaction_recovery_marker(self) -> None:
        self.transaction_recovery_path.unlink(missing_ok=True)
        self._sync_root_directory()
