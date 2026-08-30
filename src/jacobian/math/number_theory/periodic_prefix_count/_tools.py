"""Periodic union prefix count operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.number_theory.periodic_prefix_count._models import (
    MAX_PREFIX_CUTOFF_DIGITS,
    PeriodicUnionPrefixCountRequest,
    PeriodicUnionPrefixCountResult,
)
from jacobian.math.number_theory.periodic_prefix_count.operations import (
    compute_periodic_union_prefix_count,
)


def compute_periodic_union_prefix_count_op(
    request: PeriodicUnionPrefixCountRequest,
) -> PeriodicUnionPrefixCountResult:
    return compute_periodic_union_prefix_count(
        request.source, parse_canonical_integer(request.cutoff)
    )


def ppc_operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
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
    ppc_operation(
        "congruence.periodic_union.prefix_count.compute",
        "Compute the exact prefix count of a periodic congruence union",
        (
            "Given a finite periodic congruence union (optionally complemented) "
            "and a nonnegative integer cutoff of at most "
            f"{MAX_PREFIX_CUTOFF_DIGITS} digits, return the exact number of "
            "integers in [1, cutoff] that belong to the declared periodic set."
        ),
        PeriodicUnionPrefixCountRequest,
        PeriodicUnionPrefixCountResult,
        compute_periodic_union_prefix_count_op,
        "number-theory",
        "exact",
        examples=(
            example(
                "mod2_or_mod3",
                "On [1,6], the union of 0 mod 2 and 1 mod 3 has count 4.",
                {
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

__all__ = ["TOOLS"]
