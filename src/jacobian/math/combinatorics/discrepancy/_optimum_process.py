"""Bounded process owner for discrepancy minimization backends."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian.canonical import CanonicalLimits
from jacobian.math.combinatorics.discrepancy._models import (
    MAX_OPTIMUM_PROOF_MILLISECONDS,
    MAX_OPTIMUM_SOLVER_MILLISECONDS,
    DiscrepancyOptimumResult,
    FiniteSetSystem,
    _budget_exceeded_result,
    _execution_failed_result,
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
_WORKER_OUTPUT_BYTES = CanonicalLimits().max_output_bytes
_WORKER_ERROR_BYTES = 16_384
_WORKER_ADDRESS_SPACE_BYTES = 1_536 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024


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
                return _budget_exceeded_result(set_system)
            completed = run_bounded_process(
                [sys.executable, str(_OPTIMUM_WORKER)],
                input_bytes=payload,
                timeout_seconds=remaining_seconds,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_WORKER_OUTPUT_BYTES,
                stderr_limit=_WORKER_ERROR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=math.ceil(_OPTIMUM_WORKER_WALL_SECONDS),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=directory,
            )
    except OSError:
        return _execution_failed_result(set_system)
    if completed.timed_out or time.monotonic() >= deadline:
        return _budget_exceeded_result(set_system)
    if (
        completed.cancelled
        or completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        return _execution_failed_result(set_system)
    try:
        result = DiscrepancyOptimumResult.model_validate(
            json.loads(completed.stdout.decode("utf-8"))
        )
        if result.set_system != set_system:
            raise ValueError("worker result is not bound to the submitted set system")
        return result
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return _execution_failed_result(set_system)
