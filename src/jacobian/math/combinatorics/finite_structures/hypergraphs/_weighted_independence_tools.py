"""Weighted independent-selection declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.finite_structures.hypergraphs._weighted_independence import (
    WeightedIndependentSelectionRequest,
    WeightedIndependentSelectionResult,
    maximum_weight_independent_selection,
)

WEIGHTED_INDEPENDENT_SELECTION_OPERATION = MathTool(
    operation_id="hypergraph.independent_selection.rational_weight.maximum.compute",
    title="Maximize rational weight of an independent vertex selection",
    description="Return an exact optimum or an attaining incumbent with rigorous source-derived rational objective bounds.",
    request_type=WeightedIndependentSelectionRequest,
    result_type=WeightedIndependentSelectionResult,
    run=maximum_weight_independent_selection,
    tags=("hypergraph", "independent-set", "rational-weight", "optimization"),
    examples=(
        OperationExample(
            name="weighted_forbidden_pair",
            description="Choose the heavier endpoint of one forbidden pair.",
            input={
                "hypergraph": {
                    "vertices": ["a", "b"],
                    "edges": [["pair", ["a", "b"]]],
                },
                "weights": [{"num": "1", "den": "1"}, {"num": "3", "den": "2"}],
            },
        ),
    ),
)

__all__ = ["WEIGHTED_INDEPENDENT_SELECTION_OPERATION"]
