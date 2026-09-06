"""Periodic union prefix count operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.number_theory.periodic_prefix_count._models import (
    MAX_PREFIX_CUTOFF_DIGITS,
    PeriodicUnionPrefixCountRequest,
    PeriodicUnionPrefixCountResult,
)
from jacobian.math.number_theory.periodic_prefix_count.operations import (
    compute_periodic_union_prefix_count,
    verify_periodic_union_prefix_count,
)


def compute_periodic_union_prefix_count_op(
    request: PeriodicUnionPrefixCountRequest,
) -> PeriodicUnionPrefixCountResult:
    return compute_periodic_union_prefix_count(request.source, request.cutoff)


TOOLS: MathTools = (
    MathTool(
        operation_id="congruence.periodic_union.prefix_count.compute",
        title="Compute the exact prefix count of a periodic congruence union",
        description=(
            "Given a finite periodic congruence union (optionally complemented) "
            "and a nonnegative integer cutoff of at most "
            f"{MAX_PREFIX_CUTOFF_DIGITS} digits, return the exact number of "
            "integers in [1, cutoff] that belong to the declared periodic set."
        ),
        request_type=PeriodicUnionPrefixCountRequest,
        result_type=PeriodicUnionPrefixCountResult,
        run=compute_periodic_union_prefix_count_op,
        tags=("number-theory", "exact"),
        examples=(
            OperationExample(
                name="mod2_or_mod3",
                description="On [1,6], the union of 0 mod 2 and 1 mod 3 has count 4.",
                input={
                    "source": {
                        "subsets": [
                            {"modulus": "2", "residues": ["0"]},
                            {"modulus": "3", "residues": ["1"]},
                        ],
                        "complement": False,
                    },
                    "cutoff": "6",
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS", "verify_periodic_union_prefix_count"]
