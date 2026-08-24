"""Nonlinear binary code operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.code_nonlinear._models import (
    BinaryCodeRequest,
    ConstantWeightProfileRequest,
    ConstantWeightProfileResult,
    ConstantWeightRequest,
    ConstantWeightResult,
    DistanceProfileResult,
    ExplicitProfileRequest,
    ExplicitProfileResult,
    ToSetSystemRequest,
    ToSetSystemResult,
    WordDistanceRequest,
    WordDistanceResult,
)
from jacobian.math.code_nonlinear._operations import (
    compute_constant_weight,
    compute_constant_weight_profile,
    compute_distance_profile,
    compute_explicit_profile,
    compute_to_set_system,
    compute_word_distance,
)


def _op[RequestT: StrictModel, ResultT: StrictModel](
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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "code.nonlinear.distance_profile.compute",
        "Compute the distance profile of a binary code",
        "Compute the minimum Hamming distance and weight profile of a canonical explicit binary code by exact enumeration. The source-bound code may be empty or degenerate; pair-work admission is checked before execution.",
        BinaryCodeRequest,
        DistanceProfileResult,
        compute_distance_profile,
        "code",
        "distance",
        "exact",
        examples=(
            example(
                "binary_code",
                "Distance profile of a simple binary code.",
                {"code": {"length": 3, "codewords": [[0, 0, 0], [1, 1, 0], [0, 1, 1]]}},
            ),
        ),
        version="2",
    ),
    _op(
        "code.nonlinear.constant_weight.compute",
        "Generate all constant-weight binary words",
        "Generate all binary words of given length and Hamming weight.",
        ConstantWeightRequest,
        ConstantWeightResult,
        compute_constant_weight,
        "code",
        "constant-weight",
        "exact",
        examples=(
            example(
                "weight_two_length_four",
                "All weight-2 binary words of length 4.",
                {"length": 4, "weight": 2},
            ),
        ),
        version="2",
    ),
    _op(
        "code.binary.word_distance.compute",
        "Compute Hamming distance between two binary words",
        "Compute the exact Hamming distance, differing coordinates, weights, and support intersection of two equal-length binary words.",
        WordDistanceRequest,
        WordDistanceResult,
        compute_word_distance,
        "code",
        "distance",
        "exact",
        examples=(
            example(
                "word_distance_01",
                "Hamming distance between [1,0,1] and [1,1,0] with differing coordinates and support intersection.",
                {"word1": [1, 0, 1], "word2": [1, 1, 0]},
            ),
        ),
    ),
    _op(
        "code.binary.explicit.profile.compute",
        "Compute the complete profile of an explicit binary code",
        "Compute length, cardinality, weight distribution, minimum/maximum pairwise Hamming distance, distance histogram, and extremal pairs for a canonical explicit binary code. Empty and singleton codes are valid and report zero pairwise distance with no witness.",
        ExplicitProfileRequest,
        ExplicitProfileResult,
        compute_explicit_profile,
        "code",
        "distance",
        "exact",
        examples=(
            example(
                "explicit_profile_three",
                "Complete distance profile of three-word code [[0,0,0],[1,1,0],[0,1,1]] with pairwise distances.",
                {"code": {"length": 3, "codewords": [[0, 0, 0], [1, 1, 0], [0, 1, 1]]}},
            ),
        ),
        version="2",
    ),
    _op(
        "code.binary.constant_weight.profile.compute",
        "Profile of a constant-weight binary code",
        "Compute the profile of a canonical explicit constant-weight binary code using support-intersection distances. Empty and singleton codes are valid; profile admission bounds pair work and exact output size.",
        ConstantWeightProfileRequest,
        ConstantWeightProfileResult,
        compute_constant_weight_profile,
        "code",
        "constant-weight",
        "exact",
        examples=(
            example(
                "const_weight_profile",
                "Profile of constant-weight code [[1,1,0,0],[1,0,1,0]] with distance via support intersection.",
                {"code": {"length": 4, "codewords": [[1, 1, 0, 0], [1, 0, 1, 0]]}},
            ),
        ),
        version="2",
    ),
    _op(
        "code.binary.explicit.to_set_system.compute",
        "Map codewords to support subsets",
        "Map each word in a canonical explicit binary code to its support subset on coordinate labels. Empty and degenerate codes are valid; MCP execution checks the retained-source and support-output bound.",
        ToSetSystemRequest,
        ToSetSystemResult,
        compute_to_set_system,
        "code",
        "set-system",
        "exact",
        examples=(
            example(
                "to_set_system_two",
                "Support subsets for two codewords [[1,0,1,0],[0,1,0,1]] on four coordinates.",
                {"code": {"length": 4, "codewords": [[1, 0, 1, 0], [0, 1, 0, 1]]}},
            ),
        ),
        version="2",
    ),
)

__all__ = ["TOOLS"]
