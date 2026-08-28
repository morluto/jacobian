"""Exact discrepancy theory operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.discrepancy._models import (
    DiscrepancyEvalRequest,
    DiscrepancyEvalResult,
    DiscrepancyOptimumRequest,
    DiscrepancyOptimumResult,
    HardConstraintRoundingRequest,
    HardConstraintRoundingResult,
)
from jacobian.math.combinatorics.discrepancy._operations import (
    _compute_optimal_discrepancy_isolated,
    compute_discrepancy,
    compute_hard_constraint_rounding,
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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    discrepancy_theory_operation(
        "discrepancy.hard_constraint_round.compute",
        "Round a rational vector under hard cardinality constraints",
        "Compute a deterministic exact binary rounding that preserves every "
        "disjoint integral hard-row sum and bounds every monitored zero-one "
        "column's rounded-minus-source error by 4d, where d is the maximum "
        "source column incidence of one coordinate.",
        HardConstraintRoundingRequest,
        HardConstraintRoundingResult,
        compute_hard_constraint_rounding,
        "discrepancy",
        "rounding",
        "set-system",
        "hard-constraint",
        "exact",
        examples=(
            example(
                "two_hard_rows",
                "Round four half-integral coordinates while preserving two "
                "row sums and ledgering two monitored-column errors; rows must "
                "partition the coordinate axis and have integral source sums.",
                {
                    "source": {
                        "coordinate_labels": ["a", "b", "c", "d"],
                        "values": [
                            {"num": "1", "den": "2"},
                            {"num": "1", "den": "2"},
                            {"num": "1", "den": "2"},
                            {"num": "1", "den": "2"},
                        ],
                        "rows": [
                            {"label": "left", "coordinates": [0, 1]},
                            {"label": "right", "coordinates": [2, 3]},
                        ],
                        "columns": [
                            {"label": "diagonal", "coordinates": [0, 2]},
                            {"label": "off_diagonal", "coordinates": [1, 3]},
                        ],
                    }
                },
            ),
        ),
    ),
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
        "Minimize the maximum absolute set imbalance over all +1/-1 "
        "colorings of a finite set system: a bounded HiGHS MILP search "
        "produces the incumbent coloring and an exact pseudo-boolean "
        "feasibility proof re-establishes minimality before OPTIMAL carries "
        "it; BUDGET_EXCEEDED and EXECUTION_FAILED make no mathematical "
        "claim.",
        DiscrepancyOptimumRequest,
        DiscrepancyOptimumResult,
        _compute_optimal_discrepancy_isolated,
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


__all__ = ["TOOLS"]
