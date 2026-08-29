"""Nonlinear binary code operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.combinatorics.codes.nonlinear._budget import (
    require_set_system_output_bound,
)
from jacobian.math.combinatorics.codes.nonlinear._models import (
    ConstantWeightProfileRequest,
    ConstantWeightProfileResult,
    ConstantWeightRequest,
    ConstantWeightResult,
    ExplicitProfileRequest,
    ExplicitProfileResult,
    ToSetSystemRequest,
    ToSetSystemResult,
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


def _to_set_system(request: ToSetSystemRequest) -> ToSetSystemResult:
    try:
        require_set_system_output_bound(request.code)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("code",),
            code="nonlinear_code.set_system_not_admitted",
            message=str(exc),
        ) from exc
    return to_set_system(request.code)


def _op[RequestT: StrictModel, ResultT: StrictModel](
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
    _op(
        "code.nonlinear.constant_weight.compute",
        "Generate all constant-weight binary words",
        "Generate all binary words of given length and Hamming weight; the exact work and result are bounded by length times binomial(length, weight).",
        ConstantWeightRequest,
        ConstantWeightResult,
        _constant_weight,
        "code",
        "constant-weight",
        "exact",
        examples=(
            example(
                "weight_two_length_four",
                "Generate every weight-2 binary word of length 4; weight must not exceed length.",
                {"length": 4, "weight": 2},
            ),
        ),
    ),
    _op(
        "code.binary.word_distance.compute",
        "Compute Hamming distance between two binary words",
        "Compute the exact Hamming distance, differing coordinates, weights, and support intersection of two equal-length binary words; the exact result is bounded by the retained words plus their actual differing coordinates.",
        WordDistanceRequest,
        WordDistanceResult,
        _word_distance,
        "code",
        "distance",
        "exact",
        examples=(
            example(
                "word_distance_01",
                "Compute the Hamming relation between [1,0,1] and [1,1,0]; both words must be nonempty and have equal length.",
                {"word1": [1, 0, 1], "word2": [1, 1, 0]},
            ),
        ),
    ),
    _op(
        "code.binary.explicit.profile.compute",
        "Compute the complete profile of an explicit binary code",
        "Compute retained source metadata, complete weight and distance histograms, pair accounting, and compact extremal word-pair witnesses without materializing a distance graph.",
        ExplicitProfileRequest,
        ExplicitProfileResult,
        _explicit_profile,
        "code",
        "distance",
        "exact",
        examples=(
            example(
                "explicit_profile_three",
                "Compute the complete compact profile of a three-word code; the canonical source declares its ambient length and contains distinct binary words.",
                {
                    "code": {
                        "length": 3,
                        "codewords": [[0, 0, 0], [1, 1, 0], [0, 1, 1]],
                    }
                },
            ),
        ),
    ),
    _op(
        "code.binary.constant_weight.profile.compute",
        "Profile of a constant-weight binary code",
        "Compute complete distance and support-intersection histograms with source-bound extremal witnesses for a nonempty constant-weight explicit code.",
        ConstantWeightProfileRequest,
        ConstantWeightProfileResult,
        _constant_weight_profile,
        "code",
        "constant-weight",
        "exact",
        examples=(
            example(
                "const_weight_profile",
                "Compute the distance/intersection profile of two weight-2 words; the canonical source must be nonempty and every word must have the same weight.",
                {
                    "code": {
                        "length": 4,
                        "codewords": [[1, 1, 0, 0], [1, 0, 1, 0]],
                    }
                },
            ),
        ),
    ),
    _op(
        "code.binary.explicit.to_set_system.compute",
        "Map codewords to support subsets",
        "Map each canonical source codeword to its exact support block on the retained coordinate axis.",
        ToSetSystemRequest,
        ToSetSystemResult,
        _to_set_system,
        "code",
        "set-system",
        "exact",
        examples=(
            example(
                "to_set_system_two",
                "Convert two length-four codewords to support blocks.",
                {
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
