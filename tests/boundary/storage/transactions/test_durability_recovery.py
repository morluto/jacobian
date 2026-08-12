from __future__ import annotations

import os
import sqlite3
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from jacobian.storage.errors import ArtifactNotFoundError, StorageError
from jacobian.storage.repository import ArtifactRepository
from jacobian.storage.transactions import ArtifactTransactions, transaction_active_for


def test_metadata_connections_use_full_synchronous_durability(tmp_path: Path) -> None:
    store = ArtifactRepository(tmp_path)
    with store.connection() as connection:
        synchronous = connection.execute("PRAGMA synchronous").fetchone()
    assert synchronous is not None
    assert synchronous[0] == 2


@pytest.mark.skipif(os.name == "nt", reason="Windows has no directory fsync")
def test_transaction_batches_blob_directory_syncs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactRepository(tmp_path)
    payloads_by_prefix: dict[str, list[bytes]] = {}
    value = 0
    while not any(len(payloads) == 2 for payloads in payloads_by_prefix.values()):
        payload = value.to_bytes(2, "big")
        prefix = sha256(payload).hexdigest()[:2]
        payloads_by_prefix.setdefault(prefix, []).append(payload)
        value += 1
    payloads = next(
        payloads for payloads in payloads_by_prefix.values() if len(payloads) == 2
    )
    synced: list[Path] = []
    real_sync_directory = store._transactions.sync_directory

    def record_sync(directory: Path) -> None:
        if directory == store.root:
            real_sync_directory(directory)
            return
        synced.append(directory)
        real_sync_directory(directory)

    monkeypatch.setattr(store._transactions, "sync_directory", record_sync)
    with store.transaction():
        first_digest = store._blobs.write(payloads[0])
        store._blobs.write(payloads[1])
        assert synced == []
        assert store.transaction_recovery_path.exists()

    assert synced == [store._blobs.blob_path(first_digest).parent, store.blob_root]


@pytest.mark.skipif(os.name == "nt", reason="Windows has no directory fsync")
def test_transaction_body_failure_flushes_before_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactRepository(tmp_path)
    real_sync_directory = store._transactions.sync_directory
    synced: list[Path] = []

    def record_sync(directory: Path) -> None:
        if directory != store.root:
            synced.append(directory)
        real_sync_directory(directory)

    monkeypatch.setattr(store._transactions, "sync_directory", record_sync)
    descriptor_uri = ""
    with (
        pytest.raises(RuntimeError, match="abort transaction"),
        store.transaction(),
    ):
        descriptor_uri = store.register_descriptor(
            kind="schema",
            name="example.body-failure",
            version="1",
            definition={"type": "object"},
        )
        raise RuntimeError("abort transaction")

    assert synced
    assert synced[-1] == store.blob_root
    assert not store.transaction_recovery_path.exists()
    with pytest.raises(ArtifactNotFoundError):
        store.get_descriptor(descriptor_uri, expected_kind="schema")


@pytest.mark.skipif(os.name == "nt", reason="Windows has no directory fsync")
def test_persistent_directory_sync_failure_poisoned_until_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactRepository(tmp_path)
    real_sync_directory = store._transactions.sync_directory

    def fail_blob_prefix_sync(directory: Path) -> None:
        if directory.parent == store.blob_root:
            raise OSError("persistent directory sync failure")
        real_sync_directory(directory)

    monkeypatch.setattr(store._transactions, "sync_directory", fail_blob_prefix_sync)
    with (
        pytest.raises(StorageError, match="reopen the store to recover"),
        store.transaction(),
    ):
        store.register_descriptor(
            kind="schema",
            name="example.persistent-sync-failure",
            version="1",
            definition={"type": "object"},
        )

    assert store.transaction_recovery_path.exists()
    assert not store.transaction_active
    assert not transaction_active_for(store.db_path)
    with pytest.raises(StorageError, match="requires recovery"):
        store.register_descriptor(
            kind="schema",
            name="example.poisoned",
            version="1",
            definition={"type": "object"},
        )
    recovered = ArtifactRepository(tmp_path)
    assert not recovered.transaction_recovery_path.exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows has no directory fsync")
def test_recovery_sync_failure_leaves_marker_for_later_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactRepository(tmp_path)
    real_sync_directory = ArtifactTransactions.sync_directory

    def fail_blob_prefix_sync(self: ArtifactTransactions, directory: Path) -> None:
        if directory.parent == self.blob_root:
            raise OSError("persistent recovery sync failure")
        real_sync_directory(self, directory)

    monkeypatch.setattr(ArtifactTransactions, "sync_directory", fail_blob_prefix_sync)
    with (
        pytest.raises(StorageError, match="reopen the store to recover"),
        store.transaction(),
    ):
        store.register_descriptor(
            kind="schema",
            name="example.failed-publication-sync",
            version="1",
            definition={"type": "object"},
        )
    assert store.transaction_recovery_path.exists()
    with pytest.raises(OSError, match="persistent recovery sync failure"):
        ArtifactRepository(tmp_path)
    assert store.transaction_recovery_path.exists()
    monkeypatch.setattr(ArtifactTransactions, "sync_directory", real_sync_directory)
    assert not ArtifactRepository(tmp_path).transaction_recovery_path.exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows has no directory fsync")
def test_preopened_store_recovers_existing_marker_before_new_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed_store = ArtifactRepository(tmp_path)
    preopened_store = ArtifactRepository(tmp_path)
    real_failed_sync = failed_store._transactions.sync_directory

    def fail_blob_prefix_sync(directory: Path) -> None:
        if directory.parent == failed_store.blob_root:
            raise OSError("publication sync failure")
        real_failed_sync(directory)

    monkeypatch.setattr(
        failed_store._transactions, "sync_directory", fail_blob_prefix_sync
    )
    with (
        pytest.raises(StorageError, match="reopen the store to recover"),
        failed_store.transaction(),
    ):
        failed_store.register_descriptor(
            kind="schema",
            name="example.preopened-failed",
            version="1",
            definition={"type": "object"},
        )
    assert failed_store.transaction_recovery_path.exists()

    real_write_marker = preopened_store._transactions.write_recovery_marker
    recovered_before_new_marker = False

    def assert_recovered_before_new_marker() -> None:
        nonlocal recovered_before_new_marker
        recovered_before_new_marker = (
            not preopened_store.transaction_recovery_path.exists()
        )
        real_write_marker()

    monkeypatch.setattr(
        preopened_store._transactions,
        "write_recovery_marker",
        assert_recovered_before_new_marker,
    )
    with preopened_store.transaction():
        descriptor_uri = preopened_store.register_descriptor(
            kind="schema",
            name="example.preopened-recovered",
            version="1",
            definition={"type": "object"},
        )
    assert recovered_before_new_marker
    assert (
        preopened_store.get_descriptor(descriptor_uri, expected_kind="schema")["name"]
        == "example.preopened-recovered"
    )


@pytest.mark.skipif(os.name == "nt", reason="Windows has no directory fsync")
def test_preopened_store_recovers_before_direct_repeated_descriptor_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed_store = ArtifactRepository(tmp_path)
    preopened_store = ArtifactRepository(tmp_path)
    real_failed_sync = failed_store._transactions.sync_directory

    def fail_blob_prefix_sync(directory: Path) -> None:
        if directory.parent == failed_store.blob_root:
            raise OSError("publication sync failure")
        real_failed_sync(directory)

    monkeypatch.setattr(
        failed_store._transactions, "sync_directory", fail_blob_prefix_sync
    )
    failed_uri = ""
    with (
        pytest.raises(StorageError, match="reopen the store to recover"),
        failed_store.transaction(),
    ):
        failed_uri = failed_store.register_descriptor(
            kind="schema",
            name="example.preopened-direct-repeat",
            version="1",
            definition={"type": "object"},
        )
    assert failed_store.transaction_recovery_path.exists()

    recovery_sync_seen = False
    real_preopened_sync = preopened_store._transactions.sync_directory
    real_remove_marker = preopened_store._transactions.remove_recovery_marker

    def observe_recovery_sync(directory: Path) -> None:
        nonlocal recovery_sync_seen
        if directory.parent == preopened_store.blob_root:
            recovery_sync_seen = True
        real_preopened_sync(directory)

    def assert_synced_before_marker_removal() -> None:
        assert recovery_sync_seen
        real_remove_marker()

    monkeypatch.setattr(
        preopened_store._transactions, "sync_directory", observe_recovery_sync
    )
    monkeypatch.setattr(
        preopened_store._transactions,
        "remove_recovery_marker",
        assert_synced_before_marker_removal,
    )
    repeated_uri = preopened_store.register_descriptor(
        kind="schema",
        name="example.preopened-direct-repeat",
        version="1",
        definition={"type": "object"},
    )
    assert repeated_uri == failed_uri
    assert recovery_sync_seen


@pytest.mark.parametrize("failure", ["rollback", "close"])
def test_cleanup_failure_clears_ownership_and_defers_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    store = ArtifactRepository(tmp_path)
    real_connect = store.database.connect
    connection = real_connect()
    closed = False

    class CleanupFailureConnection:
        def __getattr__(self, name: str) -> Any:
            return getattr(connection, name)

        def rollback(self) -> None:
            if failure == "rollback":
                raise sqlite3.OperationalError("simulated rollback failure")
            connection.rollback()

        def close(self) -> None:
            nonlocal closed
            closed = True
            connection.close()
            if failure == "close":
                raise sqlite3.OperationalError("simulated close failure")

    monkeypatch.setattr(
        store.database,
        "connect",
        lambda: CleanupFailureConnection(),
    )
    with (
        pytest.raises(StorageError, match="cleanup was not durable"),
        store.transaction(),
    ):
        raise RuntimeError("abort transaction")
    assert closed
    assert not store.transaction_active
    assert not transaction_active_for(store.db_path)
    assert store.transaction_recovery_path.exists()
    monkeypatch.setattr(store.database, "connect", real_connect)
    with pytest.raises(StorageError, match="requires recovery"):
        store.find_by_object_digest("sha256:" + "0" * 64)
    assert not ArtifactRepository(tmp_path).transaction_recovery_path.exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows has no directory fsync")
def test_markerless_direct_sync_failure_is_synced_before_quota_clearance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactRepository(tmp_path)
    real_store_sync = store._transactions.sync_directory

    def fail_blob_prefix_sync(directory: Path) -> None:
        if directory.parent == store.blob_root:
            raise OSError("direct publication sync failure")
        real_store_sync(directory)

    monkeypatch.setattr(store._transactions, "sync_directory", fail_blob_prefix_sync)
    with pytest.raises(StorageError, match="could not write artifact data"):
        store.register_descriptor(
            kind="schema",
            name="example.markerless-sync-failure",
            version="1",
            definition={"type": "object"},
        )
    assert not store.transaction_recovery_path.exists()
    with sqlite3.connect(store.db_path) as connection:
        required_before_reopen = connection.execute(
            "SELECT reconciliation_required FROM blob_quota WHERE id = 0"
        ).fetchone()
    assert required_before_reopen == (1,)

    real_sync_directory = ArtifactTransactions.sync_directory
    observed_prefix_sync = False

    def assert_quota_still_requires_recovery(
        self: ArtifactTransactions, directory: Path
    ) -> None:
        nonlocal observed_prefix_sync
        if directory.parent == self.blob_root:
            with sqlite3.connect(self.db_path) as connection:
                row = connection.execute(
                    "SELECT reconciliation_required FROM blob_quota WHERE id = 0"
                ).fetchone()
            assert row == (1,)
            observed_prefix_sync = True
        real_sync_directory(self, directory)

    monkeypatch.setattr(
        ArtifactTransactions, "sync_directory", assert_quota_still_requires_recovery
    )
    recovered = ArtifactRepository(tmp_path)
    assert observed_prefix_sync
    assert recovered._blobs.blob_bytes_committed() > 0
    with recovered.connection() as connection:
        required_after_reopen = connection.execute(
            "SELECT reconciliation_required FROM blob_quota WHERE id = 0"
        ).fetchone()
    assert required_after_reopen is not None
    assert required_after_reopen[0] == 0
