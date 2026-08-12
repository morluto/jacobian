"""Executable correctness contract for the disposable #1224 storage spike."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from benchmarks.tooling.command_runner import (
    ToolCommandRequest,
    ToolCommandStatus,
    run_tool_command,
)


def test_sqlite_blob_spike_proves_correctness_before_performance() -> None:
    root = Path(__file__).parents[4]
    completed = run_tool_command(
        ToolCommandRequest(
            executable=sys.executable,
            arguments=(
                "tools/benchmark_storage_blobs.py",
                "--sizes",
                "1024",
                "--concurrency",
                "1",
                "--iterations",
                "1",
            ),
            cwd=str(root),
            timeout_seconds=30,
            stdout_limit_bytes=1024 * 1024,
            stderr_limit_bytes=1024 * 1024,
        )
    )
    assert completed.status is ToolCommandStatus.EXITED
    assert completed.exit_code == 0, completed.stderr.decode(errors="replace")
    report = json.loads(completed.stdout)

    assert report["warning"] == "experiment only; not a production storage backend"
    assert report["sqlite_correctness"] == {
        "backup_restore": True,
        "bounded_read": True,
        "restart": True,
        "rollback": True,
    }
    assert {
        (sample["backend"], sample["operation"]) for sample in report["samples"]
    } == {
        (backend, operation)
        for backend in ("filesystem_cas", "sqlite_blob")
        for operation in ("unique_write", "verified_read", "deduplicated_write")
    }
