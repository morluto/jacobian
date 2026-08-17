"""Exact discrepancy theory operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.discrepancy_theory._models import (
    DiscrepancyEvalRequest,
    DiscrepancyEvalResult,
    DiscrepancyOptimumRequest,
    DiscrepancyOptimumResult,
)
from jacobian.math.discrepancy_theory._operations import (
    compute_discrepancy,
    compute_optimal_discrepancy,
)


def discrepancy_theory_operation[
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
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


DISCREPANCY_THEORY_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    discrepancy_theory_operation(
        "discrepancy.theory.eval.compute",
        "Evaluate the discrepancy of a coloring on a finite set system",
        "Compute the signed sum on every set and the maximum absolute "
        "imbalance for a given +1/-1 coloring of a finite ground set.",
        DiscrepancyEvalRequest,
        DiscrepancyEvalResult,
        compute_discrepancy,
        "discrepancy",
        "set-system",
        "exact",
        examples=(
            example(
                "simple_eval",
                "Evaluate a two-element set system with coloring [+1, -1].",
                {
                    "set_system": {
                        "ground_set_size": 2,
                        "sets": [[0], [1]],
                    },
                    "coloring": [1, -1],
                },
            ),
        ),
    ),
    discrepancy_theory_operation(
        "discrepancy.theory.optimum.compute",
        "Search for a coloring minimizing maximum discrepancy",
        "Search over all 2^n colorings (bounded n <= 20) of a finite set "
        "system to find the coloring with minimum maximum discrepancy.",
        DiscrepancyOptimumRequest,
        DiscrepancyOptimumResult,
        compute_optimal_discrepancy,
        "discrepancy",
        "set-system",
        "combinatorial-search",
        "exact",
        examples=(
            example(
                "three_element_optimum",
                "Find the optimum coloring of a three-element triangle system.",
                {
                    "set_system": {
                        "ground_set_size": 3,
                        "sets": [[0, 1], [1, 2], [0, 2]],
                    },
                },
            ),
        ),
    ),
)

TOOLS = DISCREPANCY_THEORY_OPERATIONS

__all__ = ["TOOLS"]
