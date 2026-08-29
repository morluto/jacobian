"""Typed declarations for the maximum-weight antichain operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.combinatorics.posets.weighted_antichain._models import (
    WeightedAntichainRequest,
    WeightedAntichainResult,
)
from jacobian.math.combinatorics.posets.weighted_antichain.operations import (
    compute_maximum_weight_antichain,
)


def _compute(request: WeightedAntichainRequest) -> WeightedAntichainResult:
    return compute_maximum_weight_antichain(request.poset, request.weights)


TOOLS: MathTools = (
    MathTool(
        operation_id="poset.maximum_weight_antichain.compute",
        title="Compute the exact maximum-weight antichain of a finite poset",
        description=(
            "For one bounded canonical finite poset and one nonnegative "
            "rational weight on every poset element, return the exact maximum "
            "total weight of an antichain and one deterministic maximizing "
            "antichain."
        ),
        request_type=WeightedAntichainRequest,
        result_type=WeightedAntichainResult,
        run=_compute,
        tags=("poset", "antichain", "weighted", "exact"),
        examples=(
            example(
                "chain_weights",
                "Maximum weight antichain of a 3-element chain.",
                {
                    "poset": {
                        "elements": ["a", "b", "c"],
                        "strict_order_pairs": [
                            {"lower": "a", "upper": "b"},
                            {"lower": "a", "upper": "c"},
                            {"lower": "b", "upper": "c"},
                        ],
                        "cover_relations": [
                            {"lower": "a", "upper": "b"},
                            {"lower": "b", "upper": "c"},
                        ],
                        "incomparable_pairs": [],
                        "minimal_elements": ["a"],
                        "maximal_elements": ["c"],
                        "graded": True,
                        "ranks": [
                            {"element": "a", "rank": 0},
                            {"element": "b", "rank": 1},
                            {"element": "c", "rank": 2},
                        ],
                        "poset_digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                    },
                    "weights": [
                        {"num": "1", "den": "1"},
                        {"num": "3", "den": "1"},
                        {"num": "2", "den": "1"},
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
