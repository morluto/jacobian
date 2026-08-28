"""Bounded process owner for the maximum-cut Z3 acceleration."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian.math.graphs.optimization._maximum_cut import (
    MAXIMUM_CUT_RESULT_BYTES,
    GraphMaximumCutRequest,
    GraphMaximumCutResult,
    _compute_maximum_cut_without_z3,
)
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_MAXIMUM_CUT_WORKER = Path(__file__).with_name("_maximum_cut_worker.py")
_MAXIMUM_CUT_WORKER_WALL_SECONDS = 120
_WORKER_ERROR_BYTES = 16_384
_WORKER_ADDRESS_SPACE_BYTES = 1_536 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024


def compute_maximum_cut_isolated(
    request: GraphMaximumCutRequest,
) -> GraphMaximumCutResult:
    """Run Z3 outside the host and retain the admitted exact fallback."""

    deadline = time.monotonic() + _MAXIMUM_CUT_WORKER_WALL_SECONDS
    try:
        with TemporaryDirectory(prefix="jacobian-maximum-cut-") as directory:
            payload = json.dumps(
                request.model_dump(mode="json"),
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                return _compute_maximum_cut_without_z3(request)
            completed = run_bounded_process(
                [sys.executable, str(_MAXIMUM_CUT_WORKER)],
                input_bytes=payload,
                timeout_seconds=remaining_seconds,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=MAXIMUM_CUT_RESULT_BYTES,
                stderr_limit=_WORKER_ERROR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=math.ceil(_MAXIMUM_CUT_WORKER_WALL_SECONDS),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=directory,
            )
    except OSError:
        return _compute_maximum_cut_without_z3(request)
    if (
        completed.timed_out
        or completed.cancelled
        or completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
        or time.monotonic() >= deadline
    ):
        return _compute_maximum_cut_without_z3(request)
    try:
        result = GraphMaximumCutResult.model_validate(
            json.loads(completed.stdout.decode("utf-8"))
        )
        if result.graph != request.graph:
            raise ValueError("worker result is not bound to the submitted graph")
        return result
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return _compute_maximum_cut_without_z3(request)
