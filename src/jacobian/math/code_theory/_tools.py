"""Code theory operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.code_linear._models import (
    DualCodeRequest,
    DualCodeResult,
    SyndromeRequest,
    SyndromeResult,
)
from jacobian.math.code_linear._operations import (
    compute_dual_code,
    compute_syndrome,
)
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
    ct_operation(
        "code.dual_code.compute",
        "Compute the dual code",
        "Compute the exact dual code of a canonical prime-field encoder. "
        "The retained operation ID returns the shared dual encoder and "
        "parity-check value without discarding its ordered coordinate axis.",
        DualCodeRequest,
        DualCodeResult,
        compute_dual_code,
        "code",
        "dual-code",
        "exact",
        examples=(
            example(
                "hamming_7_4_generator",
                "Compute the dual code of a [7,4] Hamming encoder; preserve "
                "the encoder's ordered coordinate axis unchanged.",
                {
                    "encoder": {
                        "field_order": 2,
                        "message_axis": ["m0", "m1", "m2", "m3"],
                        "coordinate_axis": ["x0", "x1", "x2", "x3", "x4", "x5", "x6"],
                        "generator_matrix": [
                            [1, 0, 0, 0, 1, 1, 0],
                            [0, 1, 0, 0, 1, 0, 1],
                            [0, 0, 1, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1, 1, 1],
                        ],
                    }
                },
            ),
        ),
    ),
    ct_operation(
        "code.syndrome.compute",
        "Compute the syndrome of a received word",
        "Compute the exact syndrome Hw^T over GF(p) from the shared "
        "parity-check value and its matching ordered word axis.",
        SyndromeRequest,
        SyndromeResult,
        compute_syndrome,
        "code",
        "syndrome",
        "exact",
        examples=(
            example(
                "single_error_syndrome",
                "Syndrome of a single-bit error under a Hamming parity check; "
                "the word keeps the parity-check coordinate axis.",
                {
                    "parity_check": {
                        "field_order": 2,
                        "coordinate_axis": ["x0", "x1", "x2", "x3", "x4", "x5", "x6"],
                        "rows": [
                            [1, 1, 0, 1, 1, 0, 0],
                            [1, 0, 1, 1, 0, 1, 0],
                            [0, 1, 1, 1, 0, 0, 1],
                        ],
                    },
                    "coordinate_axis": ["x0", "x1", "x2", "x3", "x4", "x5", "x6"],
                    "word": [1, 0, 0, 0, 0, 0, 0],
                },
            ),
        ),
    ),
)

TOOLS = CODE_THEORY_OPERATIONS

__all__ = ["TOOLS"]
