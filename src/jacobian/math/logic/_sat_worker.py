"""Isolated Z3 adapter for one canonical SAT request."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    request_execution,
)
from jacobian._worker_errors import bind_worker_deadline, worker_execution_errors
from jacobian.math.logic._sat import SatSolveRequest, _solve_sat_kernel


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        bind_worker_deadline(payload)
        request = SatSolveRequest.model_validate(payload)
        response = _solve_sat_kernel(cnf=request.cnf, timeout_ms=request.timeout_ms)
        sys.stdout.write(json.dumps(response, separators=(",", ":")))
        return 0
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc


if __name__ == "__main__":
    with worker_execution_errors(), request_execution(time.monotonic()):
        raise SystemExit(main())
