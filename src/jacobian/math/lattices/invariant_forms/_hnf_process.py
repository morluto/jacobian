"""Bounded process owner for invariant-form graph-lattice HNF."""

from __future__ import annotations

import sys
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from jacobian._execution import (
    OperationExecutionTimeoutError,
    request_checkpoint,
)
from jacobian.process import (
    BoundedProcessResult,
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_INVARIANT_FORM_WALL_SECONDS = 3600.0
_HNF_WORKER = Path(__file__).with_name("_hnf_worker.py")
_HNF_STDERR_LIMIT = 64 * 1024


def run_hnf_worker(
    payload: bytes, *, deadline: float, stdout_limit: int
) -> BoundedProcessResult:
    """Run one invariant-form HNF worker inside the caller's deadline."""

    request_checkpoint("before graph-lattice HNF")
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "invariant-form lattice deadline expired before graph-lattice HNF"
        )
    with TemporaryDirectory(prefix="jacobian-invariant-form-hnf-") as directory:
        return run_bounded_process(
            [sys.executable, str(_HNF_WORKER)],
            input_bytes=payload,
            timeout_seconds=remaining,
            environment=worker_environment(locale="C.UTF-8"),
            stdout_limit=stdout_limit,
            stderr_limit=_HNF_STDERR_LIMIT,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=max(1, ceil(_INVARIANT_FORM_WALL_SECONDS)),
                address_space_bytes=1024 * 1024 * 1024,
                file_size_bytes=1024 * 1024,
            ),
            cwd=directory,
        )


__all__ = ["run_hnf_worker"]
