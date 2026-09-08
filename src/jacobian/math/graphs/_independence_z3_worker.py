"""Isolated Z3 adapter for one bounded graph-independence optimization."""

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
from jacobian.math.graphs._independence_z3 import (
    _solve_independence_number_values_kernel,
)
from jacobian.math.graphs.independence import IndependenceNumberBudget
from jacobian.math.graphs.values import SimpleUndirectedGraph


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        bind_worker_deadline(payload)
        if not isinstance(payload, dict):
            raise ValueError("worker payload must be an object")
        graph = SimpleUndirectedGraph.model_validate(payload["graph"])
        resource_budget = IndependenceNumberBudget.model_validate(
            payload["resource_budget"]
        )
        result = _solve_independence_number_values_kernel(graph, resource_budget)
        sys.stdout.write(
            json.dumps(
                result.model_dump(mode="json", exclude={"graph"}),
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
        return 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc


if __name__ == "__main__":
    with worker_execution_errors(), request_execution(time.monotonic()):
        raise SystemExit(main())
