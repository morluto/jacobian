"""Isolated Z3 adapter for one bounded graph-independence optimization."""

from __future__ import annotations

import json
import sys
from typing import Any

from jacobian.math.graphs._independence_z3 import (
    _solve_independence_number_values_kernel,
)
from jacobian.math.graphs.independence import IndependenceNumberBudget
from jacobian.math.graphs.values import SimpleUndirectedGraph


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
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
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
