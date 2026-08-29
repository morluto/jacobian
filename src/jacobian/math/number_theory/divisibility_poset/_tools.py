"""Operation declarations for divisibility-poset construction."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.number_theory.divisibility_poset._models import (
    DivisibilityPosetRequest,
    IntegerDivisibilityPosetResult,
)
from jacobian.math.number_theory.divisibility_poset.operations import (
    compute_divisibility_poset,
)


def _compute(
    request: DivisibilityPosetRequest,
) -> IntegerDivisibilityPosetResult:
    return compute_divisibility_poset(request.source_set)


TOOLS: MathTools = (
    MathTool(
        operation_id="number_theory.divisibility_poset.compute",
        title="Compute the divisibility poset of a positive integer set",
        description=(
            "Given a bounded finite set of distinct positive integers "
            "(at most 64 elements), construct the canonical finite poset "
            "under proper divisibility (x divides y, x != y). Each poset "
            "element carries its source integer. Generated labels are short "
            "stable identifiers so that the ElementLabel character cap never "
            "constrains source-integer digit length."
        ),
        request_type=DivisibilityPosetRequest,
        result_type=IntegerDivisibilityPosetResult,
        run=_compute,
        tags=(
            "number-theory",
            "divisibility",
            "poset",
            "partial-order",
            "exact",
        ),
        examples=(
            example(
                "divisors_of_12",
                "The set {1, 2, 3, 4, 6, 12} forms a divisibility poset.",
                {
                    "source_set": {
                        "elements": ["1", "2", "3", "4", "6", "12"],
                    },
                },
            ),
            example(
                "pair_coprime",
                "The set {2, 3} has no divisibility relations; the poset is an antichain.",
                {
                    "source_set": {
                        "elements": ["2", "3"],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
