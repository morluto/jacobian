"""Exact discrepancy theory operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.discrepancy._models import (
    DiscrepancyEvalRequest,
    DiscrepancyEvalResult,
    DiscrepancyOptimumRequest,
    DiscrepancyOptimumResult,
    HardConstraintRoundingRequest,
    HardConstraintRoundingResult,
)
from jacobian.math.combinatorics.discrepancy._optimum_process import (
    compute_optimal_discrepancy_isolated as _compute_optimal_discrepancy_isolated,
)
from jacobian.math.combinatorics.discrepancy.operations import (
    compute_discrepancy as _compute_discrepancy_native,
)
from jacobian.math.combinatorics.discrepancy.operations import (
    compute_hard_constraint_rounding as _compute_hard_constraint_rounding_native,
)
from jacobian.math.combinatorics.discrepancy.operations import (
    compute_optimal_discrepancy as _compute_optimal_discrepancy_native,
)


def compute_hard_constraint_rounding(
    request: HardConstraintRoundingRequest,
) -> HardConstraintRoundingResult:
    """Unpack a wire request for the native rounding operation."""

    return _compute_hard_constraint_rounding_native(request.source)


def compute_discrepancy(request: DiscrepancyEvalRequest) -> DiscrepancyEvalResult:
    """Unpack a wire request for the native discrepancy evaluator."""

    return _compute_discrepancy_native(request.set_system, request.coloring)


def compute_optimal_discrepancy(
    request: DiscrepancyOptimumRequest,
) -> DiscrepancyOptimumResult:
    """Unpack a wire request for the native optimum operation."""

    return _compute_optimal_discrepancy_native(request.set_system)


def compute_optimal_discrepancy_isolated(
    request: DiscrepancyOptimumRequest,
) -> DiscrepancyOptimumResult:
    """Run the native optimum operation in its killable worker process."""

    return _compute_optimal_discrepancy_isolated(request.set_system)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="discrepancy.hard_constraint_round.compute",
        title="Round a rational vector under hard cardinality constraints",
        description="Compute a deterministic exact binary rounding that preserves every "
        "disjoint integral hard-row sum and bounds every monitored zero-one "
        "column's rounded-minus-source error by 4d, where d is the maximum "
        "source column incidence of one coordinate.",
        request_type=HardConstraintRoundingRequest,
        result_type=HardConstraintRoundingResult,
        run=compute_hard_constraint_rounding,
        tags=("discrepancy", "rounding", "set-system", "hard-constraint", "exact"),
        examples=(
            OperationExample(
                name="two_hard_rows",
                description="Round four half-integral coordinates while preserving two "
                "row sums and ledgering two monitored-column errors; rows must "
                "partition the coordinate axis and have integral source sums.",
                input={
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
    MathTool(
        operation_id="discrepancy.theory.eval.compute",
        title="Evaluate the discrepancy of a coloring on a finite set system",
        description="Compute the signed sum on every set and the maximum absolute "
        "imbalance for a given +1/-1 coloring of a finite ground set.",
        request_type=DiscrepancyEvalRequest,
        result_type=DiscrepancyEvalResult,
        run=compute_discrepancy,
        tags=("discrepancy", "set-system", "exact"),
        examples=(
            OperationExample(
                name="simple_eval",
                description="Evaluate a two-element set system with coloring [+1, -1].",
                input={
                    "set_system": {
                        "ground_set_size": 2,
                        "sets": [[0], [1]],
                    },
                    "coloring": [1, -1],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="discrepancy.theory.optimum.compute",
        title="Search for a coloring minimizing maximum discrepancy",
        description="Minimize the maximum absolute set imbalance over all +1/-1 "
        "colorings of a finite set system: a bounded HiGHS MILP search "
        "produces the incumbent coloring and an exact pseudo-boolean "
        "feasibility proof re-establishes minimality before returning it. "
        "Solver exhaustion and backend failure are operational tool errors.",
        request_type=DiscrepancyOptimumRequest,
        result_type=DiscrepancyOptimumResult,
        run=compute_optimal_discrepancy_isolated,
        tags=("discrepancy", "set-system", "combinatorial-search", "exact"),
        examples=(
            OperationExample(
                name="three_element_optimum",
                description="Find the optimum coloring of a three-element triangle system.",
                input={
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
