"""Code theory operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.codes.general._models import (
    CoveringRadiusRequest,
    CoveringRadiusResult,
    LinearCodeRequest,
    MinimumDistanceResult,
    WeightDistributionResult,
)
from jacobian.math.combinatorics.codes.general.operations import (
    covering_radius,
    minimum_distance,
    weight_distribution,
)


def _minimum_distance(request: LinearCodeRequest) -> MinimumDistanceResult:
    dist = minimum_distance(request.encoder)
    return MinimumDistanceResult._from_kernel(
        request=request,
        minimum_distance=dist,
    )


def _weight_distribution(request: LinearCodeRequest) -> WeightDistributionResult:
    weights = weight_distribution(request.encoder)
    return WeightDistributionResult._from_kernel(
        request=request,
        weights=tuple((w, c) for w, c in weights),
    )


def _covering_radius(request: CoveringRadiusRequest) -> CoveringRadiusResult:
    radius = covering_radius(request.encoder)
    return CoveringRadiusResult._from_kernel(
        request=request,
        covering_radius=radius,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="code.minimum_distance.compute",
        title="Compute the minimum distance of a linear code",
        description="Compute the minimum Hamming distance by exact enumeration over a bounded prime field.",
        request_type=LinearCodeRequest,
        result_type=MinimumDistanceResult,
        run=_minimum_distance,
        tags=("code", "minimum-distance", "exact"),
        examples=(
            OperationExample(
                name="binary_repetition_code",
                description="Minimum distance of the binary repetition code of length two.",
                input={
                    "encoder": {
                        "field_order": 2,
                        "message_axis": ["m0"],
                        "coordinate_axis": ["x0", "x1"],
                        "generator_matrix": [[1, 1]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="code.weight_distribution.compute",
        title="Compute the weight distribution of a linear code",
        description="Compute the distribution of distinct codeword weights by exact enumeration over a bounded prime field.",
        request_type=LinearCodeRequest,
        result_type=WeightDistributionResult,
        run=_weight_distribution,
        tags=("code", "weight-distribution", "exact"),
        examples=(
            OperationExample(
                name="binary_repetition_code",
                description="Weight distribution of the binary repetition code of length two.",
                input={
                    "encoder": {
                        "field_order": 2,
                        "message_axis": ["m0"],
                        "coordinate_axis": ["x0", "x1"],
                        "generator_matrix": [[1, 1]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="code.covering_radius.compute",
        title="Compute the covering radius of a linear code",
        description="Compute the exact covering radius over a bounded prime field by breadth-first search on the syndrome graph.",
        request_type=CoveringRadiusRequest,
        result_type=CoveringRadiusResult,
        run=_covering_radius,
        tags=("code", "covering-radius", "exact"),
        examples=(
            OperationExample(
                name="binary_repetition_code",
                description="Covering radius of the binary repetition code of length four.",
                input={
                    "encoder": {
                        "field_order": 2,
                        "message_axis": ["m0"],
                        "coordinate_axis": ["x0", "x1", "x2", "x3"],
                        "generator_matrix": [[1, 1, 1, 1]],
                    }
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
