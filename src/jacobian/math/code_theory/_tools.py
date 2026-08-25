"""Code theory operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.code_theory._models import (
    CoveringRadiusRequest,
    CoveringRadiusResult,
    LinearCodeRequest,
    MinimumDistanceResult,
    WeightDistributionResult,
)
from jacobian.math.code_theory._operations import (
    compute_covering_radius,
    compute_min_distance,
    compute_weight_dist,
)


def ct_operation[RequestT: StrictModel, ResultT: StrictModel](
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


CODE_THEORY_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    ct_operation(
        "code.minimum_distance.compute",
        "Compute the minimum distance of a linear code",
        "Compute the minimum Hamming distance by exact enumeration over a bounded prime field.",
        LinearCodeRequest,
        MinimumDistanceResult,
        compute_min_distance,
        "code",
        "minimum-distance",
        "exact",
        examples=(
            example(
                "binary_repetition_code",
                "Minimum distance of the binary repetition code of length two.",
                {"field_order": 2, "generator_matrix": [[1, 1]]},
            ),
        ),
    ),
    ct_operation(
        "code.weight_distribution.compute",
        "Compute the weight distribution of a linear code",
        "Compute the distribution of distinct codeword weights by exact enumeration over a bounded prime field.",
        LinearCodeRequest,
        WeightDistributionResult,
        compute_weight_dist,
        "code",
        "weight-distribution",
        "exact",
        examples=(
            example(
                "binary_repetition_code",
                "Weight distribution of the binary repetition code of length two.",
                {"field_order": 2, "generator_matrix": [[1, 1]]},
            ),
        ),
    ),
    ct_operation(
        "code.covering_radius.compute",
        "Compute the covering radius of a linear code",
        "Compute the exact covering radius over a bounded prime field by breadth-first search on the syndrome graph.",
        CoveringRadiusRequest,
        CoveringRadiusResult,
        compute_covering_radius,
        "code",
        "covering-radius",
        "exact",
        examples=(
            example(
                "binary_repetition_code",
                "Covering radius of the binary repetition code of length four.",
                {"field_order": 2, "generator_matrix": [[1, 1, 1, 1]]},
            ),
        ),
    ),
)

TOOLS = CODE_THEORY_OPERATIONS

__all__ = ["TOOLS"]
