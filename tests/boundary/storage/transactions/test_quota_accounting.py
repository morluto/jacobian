from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from jacobian.canonical import CanonicalizationError
from jacobian.storage.errors import StorageError, StorageLimitError
from jacobian.storage.models import StorageLimits
from jacobian.storage.repository import ArtifactRepository


def test_over_limit_artifact_leaves_no_partial_metadata(tmp_path: Path) -> None:
    store = ArtifactRepository(
        tmp_path,
        limits=StorageLimits(
            max_artifact_bytes=2048,
            max_total_blob_bytes=1024 * 1024,
        ),
    )
    schema = store.register_descriptor(
        kind="schema",
        name="bounded.candidate",
        version="1",
        definition={"type": "object"},
    )
    semantics = store.register_descriptor(
        kind="semantics",
        name="bounded.candidate",
        version="1",
        definition={"description": "bounded fixture"},
    )
    blobs_before = {
        path.relative_to(tmp_path)
        for path in (tmp_path / "blobs" / "sha256").glob("*/*")
    }

    with pytest.raises(CanonicalizationError, match="size limit"):
        store.put(
            schema_uri=schema,
            semantics_uri=semantics,
            payload={"value": "x" * 4096},
        )

    blobs_after = {
        path.relative_to(tmp_path)
        for path in (tmp_path / "blobs" / "sha256").glob("*/*")
    }
    assert blobs_after == blobs_before


def test_concurrent_blob_commits_cannot_oversubscribe_quota(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactRepository(
        tmp_path,
        limits=StorageLimits(
            max_artifact_bytes=2048,
            max_total_blob_bytes=900,
        ),
    )
    first_accounting = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    call_lock = threading.Lock()
    accounting_calls = 0
    original_accounting = store._blobs.blob_bytes_committed

    def paused_accounting() -> int:
        nonlocal accounting_calls
        with call_lock:
            accounting_calls += 1
            call_number = accounting_calls
        if call_number == 1:
            first_accounting.set()
            assert release_first.wait(timeout=2)
        return original_accounting()

    monkeypatch.setattr(store._blobs, "blob_bytes_committed", paused_accounting)
    outcomes: list[Any] = []

    def commit(data: bytes, *, started: threading.Event | None = None) -> None:
        if started is not None:
            started.set()
        try:
            outcomes.append(store._blobs.write(data))
        except Exception as exc:
            outcomes.append(exc)

    first = threading.Thread(target=commit, args=(b"a" * 600,))
    second = threading.Thread(
        target=commit,
        args=(b"b" * 600,),
        kwargs={"started": second_started},
    )
    first.start()
    assert first_accounting.wait(timeout=1)
    second.start()
    assert second_started.wait(timeout=1)

    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    with call_lock:
        assert accounting_calls == 2
    assert sum(isinstance(outcome, str) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, StorageLimitError) for outcome in outcomes) == 1


def test_blob_writes_do_not_rescan_the_blob_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactRepository(tmp_path)

    def unexpected_scan(_path: Path) -> None:
        raise AssertionError("blob writes must use durable quota accounting")

    monkeypatch.setattr(Path, "iterdir", unexpected_scan)
    data = b"constant-time quota accounting"
    digest = store._blobs.write(data)

    assert store._blobs.blob_path(digest).read_bytes() == data
    assert store._blobs.blob_bytes_committed() == len(data)


def test_store_open_reconciles_stale_quota_metadata(tmp_path: Path) -> None:
    store = ArtifactRepository(tmp_path)
    committed = store._blobs.blob_bytes_committed()
    store._blobs.adjust_blob_bytes_committed(
        512,
        reconciliation_required=True,
    )

    reopened = ArtifactRepository(tmp_path)

    assert reopened._blobs.blob_bytes_committed() == committed


def test_store_open_migrates_legacy_quota_metadata(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    database = tmp_path / "metadata.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE blob_quota (
                id INTEGER PRIMARY KEY CHECK (id = 0),
                size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0)
            )
            """
        )
        connection.execute("INSERT INTO blob_quota (id, size_bytes) VALUES (0, 999)")

    store = ArtifactRepository(tmp_path)

    with sqlite3.connect(database) as connection:
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(blob_quota)")
        }
        row = connection.execute(
            """
            SELECT size_bytes, reconciliation_required
            FROM blob_quota
            WHERE id = 0
            """
        ).fetchone()
    assert "reconciliation_required" in columns
    assert row == (0, 0)
    assert store._blobs.blob_bytes_committed() == 0


def test_concurrent_store_open_migrates_legacy_quota_metadata_once(
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    database = tmp_path / "metadata.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE blob_quota (
                id INTEGER PRIMARY KEY CHECK (id = 0),
                size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0)
            )
            """
        )
        connection.execute("INSERT INTO blob_quota (id, size_bytes) VALUES (0, 0)")
    script = """
import sys
from jacobian.storage.repository import ArtifactRepository

ArtifactRepository(sys.argv[1])
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    completed = [process.communicate(timeout=30) for process in processes]

    assert [process.returncode for process in processes] == [0, 0], completed
    with sqlite3.connect(database) as connection:
        columns = [
            str(row[1]) for row in connection.execute("PRAGMA table_info(blob_quota)")
        ]
    assert columns.count("reconciliation_required") == 1


def test_failed_blob_publication_releases_quota_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactRepository(tmp_path)
    committed = store._blobs.blob_bytes_committed()

    def fail_link(_source: str, _target: str) -> None:
        raise OSError("link failed")

    monkeypatch.setattr(os, "link", fail_link)

    with pytest.raises(StorageError, match="could not write"):
        store._blobs.write(b"unpublished")

    assert store._blobs.blob_bytes_committed() == committed


def test_cross_process_blob_writes_cannot_oversubscribe_quota(
    tmp_path: Path,
) -> None:
    script = """
import sys
from jacobian.storage.repository import ArtifactRepository
from jacobian.storage.errors import StorageLimitError
from jacobian.storage.models import StorageLimits

store = ArtifactRepository(sys.argv[1], limits=StorageLimits(max_total_blob_bytes=900))
try:
    store._blobs.write(sys.argv[2].encode("ascii") * 600)
except StorageLimitError:
    print("limited")
else:
    print("committed")
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), value],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for value in ("a", "b")
    ]
    completed = [process.communicate(timeout=30) for process in processes]

    assert [process.returncode for process in processes] == [0, 0]
    assert sorted(stdout.strip() for stdout, _stderr in completed) == [
        "committed",
        "limited",
    ]
