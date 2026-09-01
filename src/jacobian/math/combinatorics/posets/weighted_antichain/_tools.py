"""Maximum weight antichain operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.posets.weighted_antichain._models import (
    MaximumWeightAntichainRequest,
    MaximumWeightAntichainResult,
)
from jacobian.math.combinatorics.posets.weighted_antichain.operations import (
    compute_maximum_weight_antichain,
)


def compute_mwa_op(
    request: MaximumWeightAntichainRequest,
) -> MaximumWeightAntichainResult:
    return compute_maximum_weight_antichain(request.poset, request.weights)


TOOLS: MathTools = (
    MathTool(
        operation_id="poset.maximum_weight_antichain.compute",
        title="Compute the maximum weight antichain of a poset",
        description=(
            "For one bounded canonical finite poset and one nonnegative "
            "rational weight on every poset element, return the exact "
            "maximum total weight of an antichain and one deterministic "
            "maximizing antichain."
        ),
        request_type=MaximumWeightAntichainRequest,
        result_type=MaximumWeightAntichainResult,
        run=compute_mwa_op,
        tags=("posets", "exact"),
        examples=(
            OperationExample(
                name="chain",
                description="A 3-element chain with weights 1, 2, 3.",
                input={
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
                        "poset_digest": "sha256:7505e31e11f07f0026eece8ce9621dd0dae51e613b8dfee93da02348bb80c95f",
                    },
                    "weights": [
                        {"num": "1", "den": "1"},
                        {"num": "2", "den": "1"},
                        {"num": "3", "den": "1"},
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
