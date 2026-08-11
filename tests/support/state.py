"""Per-test state and atomic immutable-template publication helpers."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from filelock import FileLock

TemplateBuilder = Callable[[Path], None]


def quiesce_sqlite_template(root: Path) -> None:
    """Checkpoint and leave a copied runtime template in rollback-journal mode.

    Runtime stores use WAL while live so readers can proceed beside writers.
    An immutable template has no concurrent writers, so publish it with a
    single durable database file.  Switching modes after the runtime closes
    also prevents a later template copy from inheriting a stale WAL policy.
    """

    import sqlite3

    database = Path(root) / "metadata.sqlite3"
    connection = sqlite3.connect(database, timeout=30)
    try:
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is None or checkpoint[0] != 0:
            raise RuntimeError(f"WAL checkpoint failed: {checkpoint!r}")
        mode = connection.execute("PRAGMA journal_mode = DELETE").fetchone()
        if mode != ("delete",):
            raise RuntimeError(f"failed to quiesce SQLite journal mode: {mode!r}")
    finally:
        connection.close()


def worker_template_target(
    tmp_path_factory: object,
    request: object,
    name: str,
) -> tuple[Path, FileLock] | None:
    """Return a run-scoped target/lock for xdist, or ``None`` locally.

    The helper accepts the pytest objects structurally so this module remains
    usable by focused tests without importing pytest itself.  The target is a
    sibling of the worker's base directory and is never created as a partial
    directory.
    """

    worker = getattr(getattr(request, "config", None), "workerinput", None)
    if not isinstance(worker, dict):
        return None
    run_id = worker.get("testrunuid")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("xdist worker did not provide a test-run identity")
    base = tmp_path_factory.getbasetemp()
    target = Path(base).parent / f"{name}-{run_id}"
    return target, FileLock(target.with_suffix(".lock"))


def publish_template(
    target: Path,
    builder: TemplateBuilder,
    *,
    lock: FileLock | None = None,
) -> Path:
    """Build ``target`` in a sibling and publish it with an atomic rename.

    A target directory is either complete and reusable or absent.  A builder
    failure removes only its staging sibling; there is no in-directory
    readiness marker that can accidentally bless a partial template.
    """

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    owned_lock = lock or FileLock(target.with_suffix(".lock"))
    with owned_lock:
        if target.exists():
            if not target.is_dir():
                raise RuntimeError(f"template target is not a directory: {target}")
            return target

        staging = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
        )
        try:
            builder(staging)
            # ``rename`` fails if a target appeared unexpectedly, preserving the
            # immutable first publication instead of replacing it.
            staging.rename(target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return target


def copy_template(template: Path, destination: Path) -> Path:
    """Copy an immutable template into a new mutable per-test directory.

    Every runtime receives independent file inodes.  In particular, blob
    files must not be hardlinked because storage integrity checks inspect
    inode metadata while another xdist worker may be copying the template.
    """

    template = Path(template)
    destination = Path(destination)
    if not template.is_dir():
        raise FileNotFoundError(f"template directory does not exist: {template}")
    if destination.exists():
        raise FileExistsError(f"mutable test state already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, destination)
    return destination
