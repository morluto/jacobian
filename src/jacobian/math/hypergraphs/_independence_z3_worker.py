"""Isolated Z3 adapter for bounded hypergraph independence kernels."""

from __future__ import annotations

import json
import sys
from typing import Any

from jacobian.math.hypergraphs._independence_z3 import (
    _solve_independence_number_kernel,
    _verify_upper_bound_kernel,
)
from jacobian.math.hypergraphs._models import (
    FiniteHypergraph,
    HypergraphIndependenceRequest,
)


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        if not isinstance(payload, dict):
            raise ValueError("worker payload must be an object")
        kind = payload["kind"]
        if kind == "solve":
            request = HypergraphIndependenceRequest.model_validate(payload["request"])
            result = _solve_independence_number_kernel(request)
            sys.stdout.write(
                json.dumps(
                    result.model_dump(
                        mode="json",
                        exclude={
                            "hypergraph",
                            "hypergraph_digest",
                            "resource_budget",
                        },
                    ),
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
            return 0
        if kind == "verify":
            hypergraph = FiniteHypergraph.model_validate(payload["hypergraph"])
            upper_bound = payload["upper_bound"]
            wall_seconds = payload["wall_seconds"]
            if (
                isinstance(upper_bound, bool)
                or not isinstance(upper_bound, int)
                or isinstance(wall_seconds, bool)
                or not isinstance(wall_seconds, int)
            ):
                raise ValueError("worker payload has invalid verification bounds")
            verified = _verify_upper_bound_kernel(hypergraph, upper_bound, wall_seconds)
            sys.stdout.write(json.dumps(verified))
            return 0
        raise ValueError("worker payload has invalid kind")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
