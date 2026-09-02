"""Bounded process owner for discrepancy minimization backends."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.math.combinatorics.discrepancy._models import (
    MAX_OPTIMUM_PROOF_MILLISECONDS,
    MAX_OPTIMUM_SOLVER_MILLISECONDS,
    DiscrepancyOptimumResult,
    FiniteSetSystem,
)
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_OPTIMUM_WORKER = Path(__file__).with_name("_optimum_worker.py")
_OPTIMUM_WORKER_WALL_SECONDS = (
    math.ceil(
        (MAX_OPTIMUM_SOLVER_MILLISECONDS + MAX_OPTIMUM_PROOF_MILLISECONDS) / 1_000
    )
    + 5
)
_WORKER_ERROR_BYTES = 16_384
_WORKER_ADDRESS_SPACE_BYTES = 1_536 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024


def _optimum_worker_stdout_limit(set_system: FiniteSetSystem) -> int:
    """Measure the largest result variant for this retained source."""

    source = set_system.model_dump(mode="json")
    projection = {
        "set_system": source,
        "optimal_coloring": [-1] * set_system.ground_set_size,
        "optimal_discrepancy": set_system.ground_set_size,
    }
    return len(json.dumps(projection, separators=(",", ":")).encode("utf-8"))


def compute_optimal_discrepancy_isolated(
    set_system: FiniteSetSystem,
) -> DiscrepancyOptimumResult:
    """Run the complete HiGHS/Z3 transaction in a bounded owner worker."""

    deadline = time.monotonic() + _OPTIMUM_WORKER_WALL_SECONDS
    try:
        with TemporaryDirectory(prefix="jacobian-discrepancy-optimum-") as directory:
            payload = json.dumps(
                set_system.model_dump(mode="json"),
                separators=(",", ":"),
            ).encode("utf-8")
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise OperationExecutionTimeoutError(
                    "discrepancy optimization deadline expired before worker startup"
                )
            completed = run_bounded_process(
                [sys.executable, str(_OPTIMUM_WORKER)],
                input_bytes=payload,
                timeout_seconds=remaining_seconds,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_optimum_worker_stdout_limit(set_system),
                stderr_limit=_WORKER_ERROR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=math.ceil(_OPTIMUM_WORKER_WALL_SECONDS),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=directory,
            )
    except OSError as exc:
        raise RuntimeError("bounded discrepancy worker could not be started") from exc
    if completed.timed_out or time.monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            "discrepancy optimization deadline expired during worker execution"
        )
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "discrepancy optimization cancelled during worker execution"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError("bounded discrepancy worker did not establish an optimum")
    try:
        result = DiscrepancyOptimumResult.model_validate(
            json.loads(completed.stdout.decode("utf-8"))
        )
        if result.set_system != set_system:
            raise ValueError("worker result is not bound to the submitted set system")
        return result
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError("bounded discrepancy worker returned malformed output") from exc
