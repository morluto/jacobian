"""Code theory operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.code_theory._dual_operations import (
    compute_dual_code,
    compute_syndrome,
)
from jacobian.math.code_theory._models import (
    CoveringRadiusRequest,
    CoveringRadiusResult,
    DualCodeRequest,
    DualCodeResult,
    LinearCodeRequest,
    MinimumDistanceResult,
    SyndromeRequest,
    SyndromeResult,
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
    ct_operation(
        "code.dual_code.compute",
        "Compute the dual code",
        "Compute the parity check matrix (dual code) from a "
        "generator matrix over a prime field GF(p), using exact null "
        "space computation.",
        DualCodeRequest,
        DualCodeResult,
        compute_dual_code,
        "code",
        "dual-code",
        "exact",
        examples=(
            example(
                "hamming_7_4_generator",
                "Compute the dual code of a [7,4] Hamming generator; "
                "field_order must be prime and entries must be canonical.",
                {
                    "field_order": 2,
                    "generator_matrix": [
                        [1, 0, 0, 0, 1, 1, 0],
                        [0, 1, 0, 0, 1, 0, 1],
                        [0, 0, 1, 0, 0, 1, 1],
                        [0, 0, 0, 1, 1, 1, 1],
                    ],
                },
            ),
        ),
    ),
    ct_operation(
        "code.syndrome.compute",
        "Compute the syndrome of a received word",
        "Compute the syndrome vector H * r^T mod p for a "
        "received word r under a parity check matrix H over GF(p).",
        SyndromeRequest,
        SyndromeResult,
        compute_syndrome,
        "code",
        "syndrome",
        "exact",
        examples=(
            example(
                "single_error_syndrome",
                "Syndrome of a single-bit error under a Hamming parity check.",
                {
                    "field_order": 2,
                    "parity_check_matrix": [
                        [1, 1, 0, 1, 1, 0, 0],
                        [1, 0, 1, 1, 0, 1, 0],
                        [0, 1, 1, 1, 0, 0, 1],
                    ],
                    "received_word": [1, 0, 0, 0, 0, 0, 0],
                },
            ),
        ),
    ),
)

TOOLS = CODE_THEORY_OPERATIONS

__all__ = ["TOOLS"]
