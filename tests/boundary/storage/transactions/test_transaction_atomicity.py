from __future__ import annotations

import threading
from pathlib import Path

import pytest

from jacobian.storage.errors import ArtifactNotFoundError
from jacobian.storage.repository import ArtifactRepository


def test_transaction_commits_multiple_descriptors_together(tmp_path: Path) -> None:
    store = ArtifactRepository(tmp_path)

    with store.transaction():
        first = store.register_descriptor(
            kind="schema",
            name="example.first",
            version="1",
            definition={"type": "object"},
        )
        second = store.register_descriptor(
            kind="semantics",
            name="example.second",
            version="1",
            definition={"description": "second"},
        )

    assert (
        store.get_descriptor(first, expected_kind="schema")["name"] == "example.first"
    )
    assert (
        store.get_descriptor(second, expected_kind="semantics")["name"]
        == "example.second"
    )
    assert not store.transaction_recovery_path.exists()


def test_transactions_serialize_across_store_instances(tmp_path: Path) -> None:
    first = ArtifactRepository(tmp_path)
    second = ArtifactRepository(tmp_path)
    writer_started = threading.Event()
    writer_entered = threading.Event()

    def write_from_second_store() -> None:
        writer_started.set()
        with second.transaction():
            writer_entered.set()
            second.register_descriptor(
                kind="schema",
                name="example.concurrent",
                version="1",
                definition={"type": "object"},
            )

    writer = threading.Thread(target=write_from_second_store)
    with first.transaction():
        first.register_descriptor(
            kind="schema",
            name="example.serialized",
            version="1",
            definition={"type": "object"},
        )
        writer.start()
        assert writer_started.wait(timeout=5)
        assert not writer_entered.wait(timeout=0.1)

    writer.join(timeout=5)
    assert not writer.is_alive()
    assert writer_entered.is_set()


def test_transaction_rolls_back_metadata_and_recovers_blob_accounting(
    tmp_path: Path,
) -> None:
    store = ArtifactRepository(tmp_path)
    descriptor_uri = ""

    with pytest.raises(RuntimeError, match="abort bootstrap"), store.transaction():
        descriptor_uri = store.register_descriptor(
            kind="schema",
            name="example.rolled-back",
            version="1",
            definition={"type": "object"},
        )
        raise RuntimeError("abort bootstrap")

    with pytest.raises(ArtifactNotFoundError):
        store.get_descriptor(descriptor_uri, expected_kind="schema")

    stored_blob_bytes = sum(
        blob.stat().st_size
        for prefix in store.blob_root.iterdir()
        if prefix.is_dir()
        for blob in prefix.iterdir()
        if blob.is_file()
    )
    assert store._blobs.blob_bytes_committed() == stored_blob_bytes

    committed = store.register_descriptor(
        kind="schema",
        name="example.after-rollback",
        version="1",
        definition={"type": "object"},
    )
    assert (
        store.get_descriptor(committed, expected_kind="schema")["name"]
        == "example.after-rollback"
    )
