"""Atomic local content-addressed artifact storage."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Self

from jacobian.canonical import CanonicalLimits
from jacobian.contracts.artifacts import ArtifactPutResult
from jacobian.persistence import (
    PersistenceLock,
    StateDatabase,
    StateDatabaseError,
)
from jacobian.persistence.migrations import (
    CURRENT_STATE_FORMAT_REVISION,
    STATE_MIGRATIONS,
    SUPPORTED_STATE_FLOOR,
)
from jacobian.storage.blobs import FilesystemBlobStore
from jacobian.storage.errors import (
    StorageCorruptionError,
    StorageError,
    UnsupportedStateVersionError,
)
from jacobian.storage.metadata import ArtifactMetadataStore
from jacobian.storage.models import StorageLimits, StoredArtifact
from jacobian.storage.transactions import ArtifactTransactions


class ArtifactRepository:
    """Content-addressed blobs plus immutable SQLite artifact metadata.

    This public aggregate owns three concrete collaborators: ``ArtifactTransactions``
    coordinates SQLite and recovery markers, ``FilesystemBlobStore`` owns the
    CAS and quota ledger, and ``ArtifactMetadataStore`` owns artifact identity
    and metadata. They are intentionally filesystem-specific collaborators,
    not a portable storage-backend abstraction.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        limits: StorageLimits | None = None,
        canonical_limits: CanonicalLimits | None = None,
        synchronous: str = "FULL",
    ) -> None:
        self.root = Path(root).resolve()
        self.limits = limits or StorageLimits()
        self.canonical_limits = canonical_limits or CanonicalLimits(
            max_input_bytes=self.limits.max_artifact_bytes,
            max_output_bytes=self.limits.max_artifact_bytes,
        )
        normalized_synchronous = synchronous.upper()
        if normalized_synchronous not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
            raise ValueError("synchronous must be one of OFF, NORMAL, FULL, or EXTRA")
        self.synchronous = normalized_synchronous
        self.blob_root = self.root / "blobs" / "sha256"
        self.staging_root = self.root / "staging"
        self.db_path = self.root / "metadata.sqlite3"
        self.blob_lock_path = self.root / ".blob-quota.lock"
        self.transaction_recovery_path = self.root / ".transaction-recovery"
        self.database = StateDatabase(self.db_path, synchronous=self.synchronous)
        self.blob_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self._transactions = ArtifactTransactions(
            blob_root=self.blob_root,
            db_path=self.db_path,
            transaction_recovery_path=self.transaction_recovery_path,
            database=self.database,
            blob_lock=PersistenceLock(self.blob_lock_path),
        )
        self._blobs = FilesystemBlobStore(
            blob_root=self.blob_root,
            staging_root=self.staging_root,
            limits=self.limits,
            transactions=self._transactions,
        )
        self._metadata = ArtifactMetadataStore(
            limits=self.limits,
            canonical_limits=self.canonical_limits,
            transactions=self._transactions,
            blobs=self._blobs,
        )
        try:
            self._reject_unsupported_state_revision()
            self.database.migrate(STATE_MIGRATIONS)
        except UnsupportedStateVersionError:
            with suppress(StateDatabaseError):
                self.database.close(checkpoint=False)
            raise
        except StorageCorruptionError:
            with suppress(StateDatabaseError):
                self.database.close(checkpoint=False)
            raise
        except Exception as exc:
            with suppress(StateDatabaseError):
                self.database.close(checkpoint=False)
            raise StorageError("artifact store schema migration failed") from exc
        self._blobs.reconcile_quota()

    def close(self) -> None:
        """Checkpoint SQLite and end this store's owned lifetime."""

        self._transactions.close()

    def __enter__(self) -> Self:
        self._transactions.ensure_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield this thread's transaction connection or one owned connection."""

        with self._transactions.connection() as connection:
            yield connection

    @property
    def transaction_active(self) -> bool:
        """Whether this thread is inside an explicit store transaction."""

        return self._transactions.transaction_active

    @property
    def transaction_identity(self) -> int | None:
        """Process-local identity of this thread's active transaction."""

        return self._transactions.transaction_identity

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit related store operations through one SQLite transaction."""

        with self._transactions.transaction(self._blobs):
            yield

    def register_descriptor(
        self,
        *,
        kind: str,
        name: str,
        version: str,
        definition: Any,
    ) -> str:
        return self._metadata.register_descriptor(
            kind=kind,
            name=name,
            version=version,
            definition=definition,
        )

    def descriptor_uri(
        self,
        *,
        kind: str,
        name: str,
        version: str,
        definition: Any,
    ) -> str:
        return self._metadata.descriptor_uri(
            kind=kind,
            name=name,
            version=version,
            definition=definition,
        )

    def get_descriptor(
        self,
        artifact_uri: str,
        *,
        expected_kind: str | None = None,
    ) -> dict[str, Any]:
        return self._metadata.get_descriptor(
            artifact_uri,
            expected_kind=expected_kind,
        )

    def put(
        self,
        *,
        schema_uri: str,
        semantics_uri: str,
        payload: Any,
        parents: tuple[str, ...] | list[str] = (),
        summary: str = "",
    ) -> ArtifactPutResult:
        return self._metadata.put(
            schema_uri=schema_uri,
            semantics_uri=semantics_uri,
            payload=payload,
            parents=parents,
            summary=summary,
        )

    def get(self, artifact_uri: str) -> StoredArtifact:
        return self._metadata.get(artifact_uri)

    def find_by_object_digest(self, object_digest: str) -> tuple[str, ...]:
        return self._metadata.find_by_object_digest(object_digest)

    def _reject_unsupported_state_revision(self) -> None:
        """Reject old and future ledgers before the migration runner can act."""

        revision = self._read_migration_revision()
        if revision is None:
            format_revision = self._read_state_format_revision()
            if format_revision is not None and (
                format_revision < SUPPORTED_STATE_FLOOR
                or format_revision > CURRENT_STATE_FORMAT_REVISION
            ):
                raise UnsupportedStateVersionError(
                    format_revision,
                    minimum_revision=SUPPORTED_STATE_FLOOR,
                )
            return
        if revision < SUPPORTED_STATE_FLOOR or revision > CURRENT_STATE_FORMAT_REVISION:
            raise UnsupportedStateVersionError(
                revision,
                minimum_revision=SUPPORTED_STATE_FLOOR,
            )
        self._validate_state_format_metadata(revision)

    def _read_migration_revision(self) -> int | None:
        if not self.db_path.exists() or self.db_path.is_dir():
            return None
        try:
            with sqlite3.connect(self.db_path) as connection:
                table = connection.execute(
                    """
                    SELECT 1 FROM sqlite_master
                    WHERE type = 'table' AND name = 'jacobian_schema_migrations'
                    """
                ).fetchone()
                if table is None:
                    return None
                row = connection.execute(
                    "SELECT MAX(revision) FROM jacobian_schema_migrations"
                ).fetchone()
        except sqlite3.DatabaseError:
            return None
        return None if row is None or row[0] is None else int(row[0])

    def _read_state_format_revision(self) -> int | None:
        if not self.db_path.exists() or self.db_path.is_dir():
            return None
        try:
            with sqlite3.connect(self.db_path) as connection:
                table = connection.execute(
                    """
                    SELECT 1 FROM sqlite_master
                    WHERE type = 'table' AND name = 'jacobian_state_format'
                    """
                ).fetchone()
                if table is None:
                    return None
                row = connection.execute(
                    """
                    SELECT format_revision
                    FROM jacobian_state_format
                    WHERE id = 0
                    """
                ).fetchone()
        except sqlite3.DatabaseError:
            return None
        return None if row is None or row[0] is None else int(row[0])

    def _validate_state_format_metadata(self, revision: int) -> None:
        try:
            with sqlite3.connect(self.db_path) as connection:
                format_table = connection.execute(
                    """
                    SELECT 1 FROM sqlite_master
                    WHERE type = 'table' AND name = 'jacobian_state_format'
                    """
                ).fetchone()
                if format_table is None:
                    return
                format_row = connection.execute(
                    """
                    SELECT format_revision
                    FROM jacobian_state_format
                    WHERE id = 0
                    """
                ).fetchone()
        except sqlite3.DatabaseError:
            return
        if format_row is None:
            if revision >= CURRENT_STATE_FORMAT_REVISION:
                raise StorageCorruptionError("state-format metadata record is missing")
            return
        format_revision = int(format_row[0])
        if format_revision < SUPPORTED_STATE_FLOOR:
            raise UnsupportedStateVersionError(
                format_revision,
                minimum_revision=SUPPORTED_STATE_FLOOR,
            )
        if format_revision > CURRENT_STATE_FORMAT_REVISION:
            raise UnsupportedStateVersionError(
                format_revision,
                minimum_revision=SUPPORTED_STATE_FLOOR,
            )
        if (
            revision >= CURRENT_STATE_FORMAT_REVISION
            and format_revision != CURRENT_STATE_FORMAT_REVISION
        ):
            raise StorageCorruptionError(
                "state-format metadata does not match the migration head"
            )
