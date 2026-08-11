"""Shared pure helpers for provider-feasibility spike scripts.

These utilities are byte-identical across every provider spike that uses
them.  They depend only on :mod:`benchmarks.tooling.command_runner` and the
standard library, so each provider ``environment/Dockerfile`` can ``COPY``
this module alongside ``command_runner.py`` into the container.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from benchmarks.tooling.command_runner import (
    ToolCommandRequest,
    ToolCommandResult,
    run_tool_command,
)

__all__ = [
    "canonical_json",
    "default_runner",
    "sha256_bytes",
]


def default_runner(
    command: Sequence[str],
    *,
    input_bytes: bytes,
    timeout_seconds: float,
    environment: Mapping[str, str],
    stdout_limit: int,
    stderr_limit: int,
) -> ToolCommandResult:
    request = ToolCommandRequest(
        executable=command[0],
        arguments=tuple(command[1:]),
        environment=environment,
        cwd=str(Path.cwd()),
        timeout_seconds=timeout_seconds,
        stdin_bytes=input_bytes,
        stdout_limit_bytes=stdout_limit,
        stderr_limit_bytes=stderr_limit,
    )
    return run_tool_command(request)


def sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
