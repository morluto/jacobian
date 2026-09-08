"""Isolated Z3 adapter for one structurally admitted SMT-LIB request."""

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
from jacobian.math.logic._smt import _solve_smt_kernel


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        bind_worker_deadline(payload)
        if not isinstance(payload, dict):
            raise ValueError("worker payload must be an object")
        logic = payload["logic"]
        smtlib = payload["smtlib"]
        timeout_ms = payload["timeout_ms"]
        if not isinstance(logic, str) or not isinstance(smtlib, str):
            raise ValueError("worker payload has invalid source fields")
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int):
            raise ValueError("worker payload has invalid timeout")
        response = _solve_smt_kernel(
            logic=logic,
            smtlib=smtlib,
            timeout_ms=timeout_ms,
        )
        sys.stdout.write(json.dumps(response, separators=(",", ":")))
        return 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc


if __name__ == "__main__":
    with worker_execution_errors(), request_execution(time.monotonic()):
        raise SystemExit(main())
