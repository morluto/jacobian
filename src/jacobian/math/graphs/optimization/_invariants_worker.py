"""Isolated Z3 adapter for the bounded clique-number operation."""

from __future__ import annotations

import json
import sys
from typing import Any

from jacobian.math.graphs.optimization._invariants import _clique_execute_kernel
from jacobian.math.graphs.optimization._models import GraphOptimizationRequest


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        request = GraphOptimizationRequest.model_validate(payload)
        result = _clique_execute_kernel(request)
        sys.stdout.write(
            json.dumps(result.model_dump(mode="json"), separators=(",", ":"))
        )
        return 0
    except (TypeError, ValueError, json.JSONDecodeError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
