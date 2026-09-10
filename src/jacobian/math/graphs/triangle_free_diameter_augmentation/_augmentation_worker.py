"""Isolated Z3 adapter for one bounded triangle-free diameter augmentation."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from jacobian._execution import request_execution
from jacobian._worker_errors import bind_worker_deadline, worker_execution_errors
from jacobian._worker_protocol import encode_worker_result_frame
from jacobian.math.graphs.triangle_free_diameter_augmentation._augmentation_z3 import (
    _solve_augmentation_kernel,
)
from jacobian.math.graphs.triangle_free_diameter_augmentation._models import (
    TriangleFreeDiameterAugmentationBudget,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        if not isinstance(payload, dict):
            raise ValueError("worker payload must be an object")
        bind_worker_deadline(payload)
        graph = SimpleUndirectedGraph.model_validate(payload["graph"])
        target_diameter = int(payload["target_diameter"])
        budget = TriangleFreeDiameterAugmentationBudget.model_validate(
            payload["resource_budget"]
        )
        admission = payload["admission"]
        if not isinstance(admission, dict):
            raise ValueError("worker admission must be an object")
        admitted = (
            tuple(admission["vertices"]),
            [tuple(edge) for edge in admission["candidates"]],
            [tuple(constraint) for constraint in admission["triangle_constraints"]],
        )
        result = _solve_augmentation_kernel(graph, target_diameter, budget, admitted)
        # Exclude graph to reduce stdout, parent will reattach
        sys.stdout.buffer.write(
            encode_worker_result_frame(
                result.model_dump(mode="json", exclude={"graph"})
            )
        )
        return 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 2


if __name__ == "__main__":
    with worker_execution_errors(), request_execution(time.monotonic()):
        raise SystemExit(main())
