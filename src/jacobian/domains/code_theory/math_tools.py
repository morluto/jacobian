"""Code theory operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.code_theory import (
    LinearCodeRequest,
    MinimumDistanceResult,
    WeightDistributionResult,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.code_theory.operations import (
    compute_min_distance,
    compute_weight_dist,
)
from jacobian.math_tools import MathTool


def ct_operation[RequestT: ContractModel, ResultT: ContractModel](
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
)
