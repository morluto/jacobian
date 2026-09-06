"""Nonlinear binary code operation declarations."""

from typing import Any

from jacobian.catalog.models import (
    MathTool,
    OperationExample,
)
from jacobian.math.combinatorics.codes.nonlinear._models import (
    ConstantWeightProfileRequest,
    ConstantWeightProfileResult,
    ConstantWeightRequest,
    ConstantWeightResult,
    ExplicitProfileRequest,
    ExplicitProfileResult,
    ToSetSystemRequest,
    WordDistanceRequest,
    WordDistanceResult,
)
from jacobian.math.combinatorics.codes.nonlinear.operations import (
    constant_weight_code,
    constant_weight_profile,
    explicit_profile,
    to_set_system,
    word_distance,
)
from jacobian.math.combinatorics.extremal_sets.values import IndexedFiniteSetFamily


def _constant_weight(request: ConstantWeightRequest) -> ConstantWeightResult:
    return constant_weight_code(request.length, request.weight)


def _word_distance(request: WordDistanceRequest) -> WordDistanceResult:
    return word_distance(request.word1, request.word2)


def _explicit_profile(request: ExplicitProfileRequest) -> ExplicitProfileResult:
    return explicit_profile(request.code)


def _constant_weight_profile(
    request: ConstantWeightProfileRequest,
) -> ConstantWeightProfileResult:
    return constant_weight_profile(request.code)


def _to_set_system(request: ToSetSystemRequest) -> IndexedFiniteSetFamily:
    return to_set_system(request.code)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="code.nonlinear.constant_weight.compute",
        title="Generate all constant-weight binary words",
        description="Generate all binary words of given length and Hamming weight; the exact work and result are bounded by length times binomial(length, weight).",
        request_type=ConstantWeightRequest,
        result_type=ConstantWeightResult,
        run=_constant_weight,
        tags=("code", "constant-weight", "exact"),
        examples=(
            OperationExample(
                name="weight_two_length_four",
                description="Generate every weight-2 binary word of length 4; weight must not exceed length.",
                input={"length": 4, "weight": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="code.binary.word_distance.compute",
        title="Compute Hamming distance between two binary words",
        description="Compute the exact Hamming distance, differing coordinates, weights, and support intersection of two equal-length binary words; the exact result is bounded by the retained words plus their actual differing coordinates.",
        request_type=WordDistanceRequest,
        result_type=WordDistanceResult,
        run=_word_distance,
        tags=("code", "distance", "exact"),
        examples=(
            OperationExample(
                name="word_distance_01",
                description="Compute the Hamming relation between [1,0,1] and [1,1,0]; both words must be nonempty and have equal length.",
                input={"word1": [1, 0, 1], "word2": [1, 1, 0]},
            ),
        ),
    ),
    MathTool(
        operation_id="code.binary.explicit.profile.compute",
        title="Compute the complete profile of an explicit binary code",
        description="Compute retained source metadata, complete weight and distance histograms, pair accounting, and compact extremal word-pair witnesses without materializing a distance graph.",
        request_type=ExplicitProfileRequest,
        result_type=ExplicitProfileResult,
        run=_explicit_profile,
        tags=("code", "distance", "exact"),
        examples=(
            OperationExample(
                name="explicit_profile_three",
                description="Compute the complete compact profile of a three-word code; the canonical source declares its ambient length and contains distinct binary words.",
                input={
                    "code": {
                        "length": 3,
                        "codewords": [[0, 0, 0], [1, 1, 0], [0, 1, 1]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="code.binary.constant_weight.profile.compute",
        title="Profile of a constant-weight binary code",
        description="Compute complete distance and support-intersection histograms with source-bound extremal witnesses for a nonempty constant-weight explicit code.",
        request_type=ConstantWeightProfileRequest,
        result_type=ConstantWeightProfileResult,
        run=_constant_weight_profile,
        tags=("code", "constant-weight", "exact"),
        examples=(
            OperationExample(
                name="const_weight_profile",
                description="Compute the distance/intersection profile of two weight-2 words; the canonical source must be nonempty and every word must have the same weight.",
                input={
                    "code": {
                        "length": 4,
                        "codewords": [[1, 1, 0, 0], [1, 0, 1, 0]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="code.binary.explicit.to_set_system.compute",
        title="Map codewords to support subsets",
        description="Map each canonical source codeword to its exact support block on the retained coordinate axis.",
        request_type=ToSetSystemRequest,
        result_type=IndexedFiniteSetFamily,
        run=_to_set_system,
        tags=("code", "set-system", "exact"),
        examples=(
            OperationExample(
                name="to_set_system_two",
                description="Convert two length-four codewords to support blocks.",
                input={
                    "code": {
                        "length": 4,
                        "codewords": [[1, 0, 1, 0], [0, 1, 0, 1]],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
