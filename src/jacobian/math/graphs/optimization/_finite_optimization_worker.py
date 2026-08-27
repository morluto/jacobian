"""Isolated Z3 adapter for one finite graph optimization operation."""

from __future__ import annotations

import json
import sys
from typing import Any

from jacobian.math.graphs.optimization._finite_optimization import _run_worker_kernel
from jacobian.math.graphs.optimization._models import GraphOptimizationRequest


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        if not isinstance(payload, dict):
            raise ValueError("worker payload must be an object")
        operation_id = payload["operation_id"]
        if not isinstance(operation_id, str):
            raise ValueError("worker payload has invalid operation id")
        request = GraphOptimizationRequest.model_validate(payload["request"])
        result = _run_worker_kernel(operation_id, request)
        sys.stdout.write(
            json.dumps(result.model_dump(mode="json"), separators=(",", ":"))
        )
        return 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
