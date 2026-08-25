"""Thin operation binding for bounded independence-number search."""

from __future__ import annotations

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.graphs.independence import (
    IndependenceNumberRequest,
    IndependenceNumberResult,
    independence_number,
)

INDEPENDENCE_NUMBER_OPERATION = MathTool(
    operation_id="graph.invariant.independence_number.compute",
    title="Independence number",
    description=(
        "Compute a maximum independent set (independence number) through order "
        "128. Return either the exact optimum or a feasible incumbent with "
        "explicit lower and upper bounds when the wall-clock budget expires."
    ),
    request_type=IndependenceNumberRequest,
    result_type=IndependenceNumberResult,
    run=independence_number,
    tags=(
        "graph",
        "invariant",
        "independent-set",
        "maximum-independent-set",
        "independence-number",
        "maximum",
        "bounded",
        "z3",
    ),
    examples=(
        example(
            "cycle_five",
            "Compute the independence number of a five-cycle.",
            {
                "graph": {
                    "vertices": ["0", "1", "2", "3", "4"],
                    "edges": [
                        ["0", "1"],
                        ["0", "4"],
                        ["1", "2"],
                        ["2", "3"],
                        ["3", "4"],
                    ],
                },
                "resource_budget": {"wall_seconds": 5, "max_order": 128},
            },
        ),
    ),
)

__all__ = ["INDEPENDENCE_NUMBER_OPERATION"]
