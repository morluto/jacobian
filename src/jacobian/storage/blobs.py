"""Content-addressed blob publication and quota accounting."""

from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from jacobian.canonical import sha256_digest
from jacobian.storage.errors import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    StorageError,
    StorageLimitError,
)
from jacobian.storage.models import StorageLimits

if TYPE_CHECKING:
    from jacobian.storage.transactions import ArtifactTransactions

_LOGGER = logging.getLogger(__name__)


class FilesystemBlobStore:
    """Publish immutable blobs and maintain crash-recoverable quota metadata."""

    def __init__(
        self,
        *,
        blob_root: Path,
        staging_root: Path,
        limits: StorageLimits,
        transactions: ArtifactTransactions,
    ) -> None:
        self._blob_root = blob_root
        self._staging_root = staging_root
        self._limits = limits
        self._transactions = transactions
        self._validated_blobs: dict[str, tuple[int, int, int, int, int]] = {}

    def blob_path(self, digest: str) -> Path:
        hex_digest = digest.removeprefix("sha256:")
        if len(hex_digest) != 64 or any(
            char not in "0123456789abcdef"  # pragma: allowlist secret
            for char in hex_digest
        ):
            raise ArtifactIntegrityError(f"invalid blob digest: {digest!r}")
        return self._blob_root / hex_digest[:2] / hex_digest[2:]

    def _scan_blob_bytes_committed(self) -> tuple[int, set[Path]]:
        total = 0
        observed_prefixes: set[Path] = set()
        for prefix in self._blob_root.iterdir():
            if not prefix.is_dir() or prefix.is_symlink():
                continue
            observed_prefixes.add(prefix)
            for blob in prefix.iterdir():
                if blob.is_file() and not blob.is_symlink():
                    digest = f"sha256:{prefix.name}{blob.name}"
                    before = blob.stat()
                    data = blob.read_bytes()
                    after = blob.stat()
                    before_signature = (
                        before.st_dev,
                        before.st_ino,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    )
                    after_signature = (
                        after.st_dev,
                        after.st_ino,
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                    )
                    if before_signature != after_signature:
                        raise ArtifactIntegrityError(
                            f"blob changed during store recovery: {digest}"
                        )
                    if sha256_digest(data) != digest:
                        raise ArtifactIntegrityError(
                            f"blob digest mismatch during store recovery: {digest}"
                        )
                    self._validated_blobs[digest] = after_signature
                    total += after.st_size
        return total, observed_prefixes

    def reconcile_quota(self, *, force: bool = False) -> None:
        """Recover quota accounting only after an interrupted blob mutation."""

        with self._exclusive_blob_lock():
            force = force or self._transactions.transaction_recovery_path.exists()
            with self._transactions.connection() as connection:
                row = connection.execute(
                    """
                    SELECT reconciliation_required
                    FROM blob_quota
                    WHERE id = 0
                    """
                ).fetchone()
                if force:
                    connection.execute(
                        """
                        UPDATE blob_quota
                        SET reconciliation_required = 1
                        WHERE id = 0
                        """
                    )
            if (
                os.name != "nt"
                and not force
                and row is not None
                and not bool(row["reconciliation_required"])
            ):
                return

            total, observed_prefixes = self._scan_blob_bytes_committed()
            for prefix in sorted(observed_prefixes, key=str):
                self._transactions.sync_directory(prefix)
            self._transactions.sync_directory(self._blob_root)
            with self._transactions.connection() as connection:
                connection.execute(
                    """
                    INSERT INTO blob_quota (
                        id,
                        size_bytes,
                        reconciliation_required
                    )
                    VALUES (0, ?, 0)
                    ON CONFLICT(id) DO UPDATE
                    SET size_bytes = excluded.size_bytes,
                        reconciliation_required = 0
                    """,
                    (total,),
                )
            if self._transactions.transaction_recovery_path.exists():
                self._transactions.remove_recovery_marker()

    def blob_bytes_committed(self) -> int:
        with self._transactions.connection() as connection:
            row = connection.execute(
                """
                SELECT size_bytes, reconciliation_required
                FROM blob_quota
                WHERE id = 0
                """
            ).fetchone()
        if row is None:
            raise ArtifactIntegrityError("artifact store quota metadata is missing")
        if bool(row["reconciliation_required"]):
            raise ArtifactIntegrityError(
                "artifact store quota metadata requires recovery"
            )
        return int(row["size_bytes"])

    def adjust_blob_bytes_committed(
        self,
        delta: int,
        *,
        reconciliation_required: bool,
    ) -> None:
        with self._transactions.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE blob_quota
                SET size_bytes = size_bytes + ?,
                    reconciliation_required = ?
                WHERE id = 0 AND size_bytes + ? >= 0
                """,
                (delta, int(reconciliation_required), delta),
            )
        if cursor.rowcount != 1:
            raise ArtifactIntegrityError("artifact store quota metadata is invalid")

    def mark_blob_quota_reconciled(self) -> None:
        with self._transactions.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE blob_quota
                SET reconciliation_required = 0
                WHERE id = 0
                """
            )
        if cursor.rowcount != 1:
            raise ArtifactIntegrityError("artifact store quota metadata is missing")

    @contextmanager
    def _exclusive_blob_lock(self) -> Iterator[None]:
        """Serialize quota accounting and blob publication across processes."""

        with self._transactions.exclusive_blob_lock():
            yield

    def write(self, data: bytes) -> str:
        try:
            return self._write_unchecked(data)
        except (OSError, sqlite3.Error) as exc:
            _LOGGER.exception("filesystem error while writing artifact data")
            raise StorageError(
                "Jacobian could not write artifact data. Check the state directory "
                "and available disk space, then retry."
            ) from exc

    def _check_existing_blob(
        self,
        target: Path,
        digest: str,
        data: bytes,
    ) -> str | None:
        """Return *digest* if *target* already stores *data*, or None if absent."""

        if not target.exists():
            return None
        if target.is_symlink() or not target.is_file():
            raise ArtifactIntegrityError(
                f"existing blob does not match digest {digest}"
            )
        stat = target.stat()
        signature = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        if self._validated_blobs.get(digest) == signature:
            return digest
        existing = target.read_bytes()
        after = target.stat()
        after_signature = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if signature != after_signature or existing != data:
            raise ArtifactIntegrityError(
                f"existing blob does not match digest {digest}"
            )
        self._validated_blobs[digest] = after_signature
        return digest

    def _publish_new_blob(
        self,
        target: Path,
        data: bytes,
        digest: str,
        *,
        prefix_created: bool,
    ) -> None:
        """Write *data* to a temp file, hard-link to *target*, and sync."""

        descriptor, temporary_name = tempfile.mkstemp(
            dir=self._staging_root,
            prefix="blob-",
        )
        temporary = Path(temporary_name)
        reserved = False
        published = False
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            self.adjust_blob_bytes_committed(len(data), reconciliation_required=True)
            reserved = True
            try:
                os.link(temporary, target)
                published = True
            except FileExistsError as exc:
                if target.is_symlink() or target.read_bytes() != data:
                    raise ArtifactIntegrityError(
                        f"concurrent blob does not match digest {digest}"
                    ) from exc
            self._transactions.sync_blob_publication_directories(
                target.parent,
                prefix_created=prefix_created,
            )
        finally:
            temporary.unlink(missing_ok=True)
            if reserved and not published:
                try:
                    self.adjust_blob_bytes_committed(
                        -len(data), reconciliation_required=False
                    )
                except (ArtifactIntegrityError, sqlite3.Error):
                    _LOGGER.exception(
                        "failed to release an unpublished blob quota reservation"
                    )
        if published:
            self.mark_blob_quota_reconciled()
        stat = target.stat()
        self._validated_blobs[digest] = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def _write_unchecked(
        self,
        data: bytes,
    ) -> str:
        digest = sha256_digest(data)
        target = self.blob_path(digest)
        with self._exclusive_blob_lock():
            if (
                not self._transactions.transaction_active
                and self._transactions.transaction_recovery_path.exists()
            ):
                self.reconcile_quota(force=True)
            prefix_created = not target.parent.exists()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.parent.is_symlink() or not target.parent.is_dir():
                raise ArtifactIntegrityError(
                    f"blob prefix is not a local directory for {digest}"
                )
            existing = self._check_existing_blob(target, digest, data)
            if existing is not None:
                return existing
            if (
                self.blob_bytes_committed() + len(data)
                > self._limits.max_total_blob_bytes
            ):
                raise StorageLimitError("artifact store blob quota would be exceeded")
            self._publish_new_blob(target, data, digest, prefix_created=prefix_created)
        return digest

    def read(self, digest: str) -> bytes:
        path = self.blob_path(digest)
        if not path.exists():
            raise ArtifactNotFoundError(f"missing blob for digest {digest}")
        if path.is_symlink() or not path.is_file():
            raise ArtifactIntegrityError(f"blob path is not a regular file: {digest}")
        data = path.read_bytes()
        if sha256_digest(data) != digest:
            raise ArtifactIntegrityError(f"blob digest mismatch: {digest}")
        return data
