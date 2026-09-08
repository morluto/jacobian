"""Isolated Z3 adapter for one SMT-core extraction or replay request."""

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
from jacobian.math.logic._unsat_core import (
    SmtUnsatCoreRequest,
    _unsat_core_worker_kernel,
)


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        bind_worker_deadline(payload)
        if not isinstance(payload, dict):
            raise ValueError("worker payload must be an object")
        request = SmtUnsatCoreRequest.model_validate(payload["request"])
        raw_selected_indices = payload.get("selected_indices")
        if raw_selected_indices is None:
            selected_indices = None
        elif isinstance(raw_selected_indices, list) and all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in raw_selected_indices
        ):
            selected_indices = tuple(raw_selected_indices)
        else:
            raise ValueError("worker payload has invalid selected indices")
        response = _unsat_core_worker_kernel(
            request,
            selected_indices=selected_indices,
        )
        sys.stdout.write(json.dumps(response, separators=(",", ":")))
        return 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc


if __name__ == "__main__":
    with worker_execution_errors(), request_execution(time.monotonic()):
        raise SystemExit(main())
