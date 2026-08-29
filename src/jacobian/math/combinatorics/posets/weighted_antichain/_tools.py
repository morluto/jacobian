"""Maximum weight antichain operation declarations."""

from collections.abc import Callable

from jacobian.catalog._examples import example
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


def mwa_action(
    operation_id: str,
    title: str,
    description: str,
    request_model: type,
    result_model: type,
    operation: Callable,
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: MathTools = (
    mwa_action(
        "poset.maximum_weight_antichain.compute",
        "Compute the maximum weight antichain of a poset",
        (
            "For one bounded canonical finite poset and one nonnegative "
            "rational weight on every poset element, return the exact "
            "maximum total weight of an antichain and one deterministic "
            "maximizing antichain."
        ),
        MaximumWeightAntichainRequest,
        MaximumWeightAntichainResult,
        compute_mwa_op,
        "posets",
        "exact",
        examples=(
            example(
                "chain",
                "A 3-element chain with weights 1, 2, 3.",
                {
                    "poset": {
                        "elements": ["a", "b", "c"],
                        "strict_order_pairs": [
                            {"lower": "a", "upper": "b"},
                            {"lower": "b", "upper": "c"},
                            {"lower": "a", "upper": "c"},
                        ],
                        "cover_relations": [
                            {"lower": "a", "upper": "b"},
                            {"lower": "b", "upper": "c"},
                        ],
                        "incomparable_pairs": [],
                        "minimal_elements": ["a"],
                        "maximal_elements": ["c"],
                        "graded": True,
                        "ranks": ["0", "1", "2"],
                        "poset_digest": "0000000000000000000000000000000000000000000000000000000000000000",
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
