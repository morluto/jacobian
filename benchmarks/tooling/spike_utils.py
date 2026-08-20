"""Shared pure helpers for provider-feasibility spike scripts.

These utilities are byte-identical across every provider spike that uses
them.  They depend only on :mod:`tools.command_runner` and the
standard library, so each provider ``environment/Dockerfile`` can ``COPY``
this module alongside ``command_runner.py`` into the container.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.command_runner import (
    ToolCommandRequest,
    ToolCommandResult,
    run_tool_command,
)

__all__ = [
    "canonical_json",
    "default_runner",
    "owned_fixture_path",
    "sha256_bytes",
]


def default_runner(
    request: ToolCommandRequest,
) -> ToolCommandResult:
    return run_tool_command(request)


def owned_fixture_path(
    module_file: str, repository_relative: str, container_name: str
) -> Path:
    """Select a passive fixture in a checkout or beside a copied container script."""

    module_path = Path(module_file).resolve()
    if len(module_path.parents) > 3:
        checkout_candidate = module_path.parents[3] / repository_relative
        if checkout_candidate.is_file():
            return checkout_candidate
    return module_path.with_name(container_name)


def sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
