"""Bounded execution helpers for task solution and oracle programs."""

from __future__ import annotations

import sys
from pathlib import Path

from tools.command_runner import (
    ToolCommandRequest,
    ToolCommandStatus,
    operator_environment,
    run_tool_command,
)

ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_LIMIT_BYTES = 64 * 1024 * 1024
_ENVIRONMENT = ("PATH", "PYTHONPATH", "VIRTUAL_ENV")


def run_solution(
    task: Path,
    app: Path,
    *,
    script: str = "solve.py",
    arguments: tuple[str, ...] = ("--root",),
    timeout_seconds: float = 300.0,
) -> None:
    """Run one task-owned producer under the bounded tooling boundary."""

    task_root = task.resolve(strict=True)
    app_root = app.resolve(strict=True)
    command_arguments = (str(task_root / "solution" / script), *arguments)
    if arguments == ("--root",):
        command_arguments = (*command_arguments, str(app_root))
    result = run_tool_command(
        ToolCommandRequest(
            executable=sys.executable,
            arguments=command_arguments,
            environment=operator_environment(include=_ENVIRONMENT),
            cwd=str(ROOT),
            timeout_seconds=timeout_seconds,
            stdout_limit_bytes=_OUTPUT_LIMIT_BYTES,
            stderr_limit_bytes=_OUTPUT_LIMIT_BYTES,
        )
    )
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        detail = result.diagnostic or result.stderr.decode("utf-8", "replace")
        raise RuntimeError(f"task producer failed: {detail.strip()}")


__all__ = ["run_solution"]
