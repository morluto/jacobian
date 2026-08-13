"""Tests for copy_template isolation."""

from pathlib import Path

import pytest
from tests.support.state import copy_template, publish_template


def test_copy_template_copies_blobs_without_sharing_inodes(tmp_path: Path) -> None:
    """Blob copies must not share inodes with the immutable template."""
    template = tmp_path / "template"
    template.mkdir()
    blob = template / "blobs" / "sha256" / "00" / "abc123"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"blob content")
    (template / "metadata.sqlite3").write_bytes(b"database")

    dest = tmp_path / "destination"
    copy_template(template, dest)

    assert (dest / "blobs" / "sha256" / "00" / "abc123").read_bytes() == b"blob content"
    template_inode = blob.stat().st_ino
    dest_inode = (dest / "blobs" / "sha256" / "00" / "abc123").stat().st_ino
    assert template_inode != dest_inode, "blob should be copied, not hardlinked"

    (dest / "blobs" / "sha256" / "00" / "abc123").write_bytes(b"changed")
    assert blob.read_bytes() == b"blob content"

    template_meta = (template / "metadata.sqlite3").stat().st_ino
    dest_meta = (dest / "metadata.sqlite3").stat().st_ino
    assert template_meta != dest_meta, "metadata should be copied, not hardlinked"


def test_copy_template_preserves_all_files(tmp_path: Path) -> None:
    """All non-blob files should be present in the destination."""
    template = tmp_path / "template"
    template.mkdir()
    (template / "blobs").mkdir()
    blob = template / "blobs" / "sha256" / "00" / "def456"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"blob")
    (template / "metadata.sqlite3").write_bytes(b"database")
    (template / "metadata.sqlite3-shm").write_bytes(b"shm")
    (template / "metadata.sqlite3-wal").write_bytes(b"wal")

    dest = tmp_path / "destination"
    copy_template(template, dest)

    for name in ("metadata.sqlite3", "metadata.sqlite3-shm", "metadata.sqlite3-wal"):
        assert (dest / name).exists(), f"{name} should be copied"


def test_copy_template_raises_on_existing_destination(tmp_path: Path) -> None:
    """copy_template should refuse to overwrite an existing destination."""
    template = tmp_path / "template"
    template.mkdir()
    (template / "blobs").mkdir()
    (template / "blobs" / "sha256" / "00" / "abc").parent.mkdir(parents=True)
    (template / "blobs" / "sha256" / "00" / "abc").write_bytes(b"blob")

    dest = tmp_path / "destination"
    dest.mkdir()
    with pytest.raises(FileExistsError):
        copy_template(template, dest)


def test_failed_template_publication_leaves_no_reusable_partial_state(
    tmp_path: Path,
) -> None:
    target = tmp_path / "template"

    def fail(staging: Path) -> None:
        (staging / "partial.sqlite3").write_text("incomplete", encoding="utf-8")
        raise RuntimeError("simulated construction failure")

    with pytest.raises(RuntimeError, match="construction failure"):
        publish_template(target, fail)

    assert not target.exists()
    assert not list(tmp_path.glob(".template.staging-*"))
