"""Isolated Z3 adapter for the bounded clique-number operation."""

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
from jacobian.math.graphs.optimization._invariants import _clique_execute_kernel
from jacobian.math.graphs.optimization._models import GraphOptimizationRequest


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        bind_worker_deadline(payload)
        request = GraphOptimizationRequest.model_validate(payload)
        result = _clique_execute_kernel(request)
        sys.stdout.write(
            json.dumps(result.model_dump(mode="json"), separators=(",", ":"))
        )
        return 0
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc


if __name__ == "__main__":
    with worker_execution_errors(), request_execution(time.monotonic()):
        raise SystemExit(main())
