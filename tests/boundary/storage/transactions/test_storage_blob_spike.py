"""Executable correctness contract for the disposable #1224 storage spike."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_sqlite_blob_spike_proves_correctness_before_performance() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "tools/benchmark_storage_blobs.py",
            "--sizes",
            "1024",
            "--concurrency",
            "1",
            "--iterations",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path(__file__).parents[4],
    )
    report = json.loads(completed.stdout)

    assert report["warning"] == "experiment only; not a production storage backend"
    assert report["sqlite_correctness"] == {
        "backup_restore": True,
        "bounded_read": True,
        "restart": True,
        "rollback": True,
    }
    assert {
        (sample["backend"], sample["operation"])
        for sample in report["samples"]
    } == {
        (backend, operation)
        for backend in ("filesystem_cas", "sqlite_blob")
        for operation in ("unique_write", "verified_read", "deduplicated_write")
    }
