"""Shared native/MCP killable process boundary for exact PB feasibility."""

import json
import math
import sys
from pathlib import Path
from time import monotonic
from typing import cast

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_checkpoint,
)
from jacobian.math.combinatorics.discrepancy.bounded_coloring._z3 import (
    BackendReply,
    BackendStatus,
    Constraints,
)
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_WORKER = Path(__file__).with_name("_worker.py")


def _failed() -> BackendReply:
    return {"status": "EXECUTION_FAILED", "coloring": None}


def decode_reply(data: bytes, variable_count: int) -> BackendReply:
    """Check the bounded worker codec without trusting a model assignment."""
    try:
        value = json.loads(data)
        if not isinstance(value, dict) or set(value) != {"status", "coloring"}:
            return _failed()
        status = value["status"]
        if status not in (
            "SATISFIABLE",
            "UNSATISFIABLE",
            "BUDGET_EXCEEDED",
            "EXECUTION_FAILED",
        ):
            return _failed()
        coloring = value["coloring"]
        if status == "SATISFIABLE":
            if not isinstance(coloring, list) or len(coloring) != variable_count:
                return _failed()
            if any(type(item) is not int or item not in (-1, 1) for item in coloring):
                return _failed()
            return {"status": "SATISFIABLE", "coloring": tuple(coloring)}
        if coloring is not None:
            return _failed()
        return {"status": cast(BackendStatus, status), "coloring": None}
    except (UnicodeDecodeError, TypeError, ValueError):
        return _failed()


def run_solver(
    variable_count: int,
    constraints: Constraints,
    work_limit: int,
    deadline: float,
    *,
    caller_limited: bool = False,
) -> BackendReply:
    request_checkpoint("before discrepancy worker startup")
    remaining = deadline - monotonic()
    if remaining <= 0:
        if caller_limited:
            raise OperationExecutionTimeoutError(
                "discrepancy decision deadline expired before the solver worker"
            )
        return {"status": "BUDGET_EXCEEDED", "coloring": None}
    payload = json.dumps(
        {
            "variable_count": variable_count,
            "constraints": constraints,
            "work_limit": work_limit,
            "deadline": deadline,
        },
        separators=(",", ":"),
    ).encode()
    try:
        completed = run_bounded_process(
            [sys.executable, str(_WORKER)],
            input_bytes=payload,
            timeout_seconds=remaining,
            environment=worker_environment(locale="C.UTF-8"),
            # At most 64 signed colours and a closed status; the source and
            # ledger stay in the owner and never multiply worker output.
            stdout_limit=1024,
            stderr_limit=16_384,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=max(1, math.ceil(remaining)),
                address_space_bytes=1536 * 1024 * 1024,
                file_size_bytes=1024 * 1024,
            ),
        )
    except OSError:
        return _failed()
    if completed.cancelled:
        raise OperationExecutionCancelledError("discrepancy decision cancelled")
    if completed.timed_out:
        if caller_limited:
            raise OperationExecutionTimeoutError(
                "discrepancy decision deadline expired during the solver worker"
            )
        return {"status": "BUDGET_EXCEEDED", "coloring": None}
    if (
        completed.returncode != 0
        or completed.stdout_exceeded
        or completed.stderr_exceeded
    ):
        return _failed()
    return decode_reply(completed.stdout, variable_count)
