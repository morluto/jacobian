"""Isolated Z3 adapter for one finite graph optimization operation."""

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
from jacobian._worker_protocol import encode_worker_result_frame
from jacobian.math.graphs.optimization._finite_optimization import _run_worker_kernel
from jacobian.math.graphs.optimization._models import GraphOptimizationRequest


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        bind_worker_deadline(payload)
        if not isinstance(payload, dict):
            raise ValueError("worker payload must be an object")
        operation_id = payload["operation_id"]
        if not isinstance(operation_id, str):
            raise ValueError("worker payload has invalid operation id")
        request = GraphOptimizationRequest.model_validate(payload["request"])
        result = _run_worker_kernel(operation_id, request)
        sys.stdout.buffer.write(
            encode_worker_result_frame(result.model_dump(mode="json"))
        )
        return 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc


if __name__ == "__main__":
    with worker_execution_errors(), request_execution(time.monotonic()):
        raise SystemExit(main())
