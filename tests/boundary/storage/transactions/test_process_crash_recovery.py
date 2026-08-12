from __future__ import annotations

import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from jacobian.storage.repository import ArtifactRepository


def test_store_open_recovers_process_death_before_blob_publication(
    tmp_path: Path,
) -> None:
    script = """
import os
import sys
from jacobian.storage.repository import ArtifactRepository

store = ArtifactRepository(sys.argv[1])
adjust = store._blobs.adjust_blob_bytes_committed

def reserve_then_exit(delta, *, reconciliation_required):
    adjust(delta, reconciliation_required=reconciliation_required)
    os._exit(0)

store._blobs.adjust_blob_bytes_committed = reserve_then_exit
store._blobs.write(b"reserved-but-unpublished")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    reopened = ArtifactRepository(tmp_path)

    assert reopened._blobs.blob_bytes_committed() == 0
    assert not tuple((tmp_path / "blobs" / "sha256").glob("*/*"))


def test_store_open_recovers_process_death_after_blob_publication(
    tmp_path: Path,
) -> None:
    data = b"published-before-clean-marker"
    script = """
import os
import sys
from jacobian.storage.repository import ArtifactRepository

store = ArtifactRepository(sys.argv[1])
store._blobs.mark_blob_quota_reconciled = lambda: os._exit(0)
store._blobs.write(sys.argv[2].encode("ascii"))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), data.decode("ascii")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    reopened = ArtifactRepository(tmp_path)
    digest = f"sha256:{sha256(data).hexdigest()}"

    assert reopened._blobs.blob_bytes_committed() == len(data)
    assert reopened._blobs.read(digest) == data


def test_store_open_recovers_process_death_during_store_transaction(
    tmp_path: Path,
) -> None:
    ArtifactRepository(tmp_path)
    script = """
import os
import sys
from jacobian.storage.repository import ArtifactRepository

store = ArtifactRepository(sys.argv[1])
with store.transaction():
    store.register_descriptor(
        kind="schema",
        name="crashed.transaction",
        version="1",
        definition={"type": "object", "description": "published before crash"},
    )
    os._exit(0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / ".transaction-recovery").is_file()

    reopened = ArtifactRepository(tmp_path)
    stored_blob_bytes = sum(
        blob.stat().st_size
        for prefix in reopened.blob_root.iterdir()
        if prefix.is_dir()
        for blob in prefix.iterdir()
        if blob.is_file()
    )

    assert reopened._blobs.blob_bytes_committed() == stored_blob_bytes
    assert not reopened.transaction_recovery_path.exists()


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows startup intentionally reconciles by scanning the blob tree",
)
def test_clean_store_open_does_not_scan_the_blob_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactRepository(tmp_path)
    data = b"validated once per unchanged blob"
    digest = store._blobs.write(data)

    def unexpected_scan(_path: Path) -> None:
        raise AssertionError("clean store startup must trust durable quota metadata")

    monkeypatch.setattr(Path, "iterdir", unexpected_scan)

    reopened = ArtifactRepository(tmp_path)

    assert reopened._blobs.blob_bytes_committed() == len(data)
    assert reopened._blobs.blob_path(digest).is_file()
