"""Isolated Z3 adapter for one bounded triangle-free diameter augmentation."""

from __future__ import annotations

import json
import sys
from typing import Any

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
        graph = SimpleUndirectedGraph.model_validate(payload["graph"])
        target_diameter = int(payload["target_diameter"])
        budget = TriangleFreeDiameterAugmentationBudget.model_validate(
            payload["resource_budget"]
        )
        result = _solve_augmentation_kernel(graph, target_diameter, budget)
        # Exclude graph to reduce stdout, parent will reattach
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
