"""Public declaration for the exact root--critical distance profile."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.root_critical._models import (
    ExactSplittingField,
    ExactSplittingFieldRequest,
    RootCriticalDistanceProfile,
    RootCriticalDistanceProfileRequest,
    SplittingFieldDistanceProfile,
    SplittingFieldDistanceRequest,
)
from jacobian.math.polynomials.root_critical.operations import (
    exact_splitting_field,
    root_critical_distance_profile,
    splitting_field_distance_profile,
)

_CUBIC_EXAMPLE = {
    "polynomial": {
        "domain": "QQ",
        "variables": ["z"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [3]},
                {"coefficient": {"num": "-1", "den": "1"}, "exponents": [0]},
            ]
        },
    }
}

# The trivial splitting field of x^2 (support x): theta = 0, defining t,
# singleton zero rectangles. Stable across backend versions.
_ZERO_FIELD = {
    "source_polynomial": {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [2]}]
        },
    },
    "squarefree_support": {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [1]}]
        },
    },
    "defining_polynomial": {
        "domain": "QQ",
        "variables": ["t"],
        "polynomial": {
            "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [1]}]
        },
    },
    "conjugation_coefficients": [{"num": "0", "den": "1"}],
    "embedding_index": 0,
    "embedding_rectangle": {
        "real_lower": {"num": "0", "den": "1"},
        "real_upper": {"num": "0", "den": "1"},
        "imaginary_lower": {"num": "0", "den": "1"},
        "imaginary_upper": {"num": "0", "den": "1"},
    },
    "roots": [
        {
            "axis_index": 0,
            "coefficients_ascending": [{"num": "0", "den": "1"}],
            "multiplicity": 1,
            "rectangle": {
                "real_lower": {"num": "0", "den": "1"},
                "real_upper": {"num": "0", "den": "1"},
                "imaginary_lower": {"num": "0", "den": "1"},
                "imaginary_upper": {"num": "0", "den": "1"},
            },
        }
    ],
}
_DOUBLE_ROOT_EXAMPLE = {
    "polynomial": {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [2]}]
        },
    },
    "splitting_field": _ZERO_FIELD,
    "max_pair_rows": 16,
}


def compute_root_critical_distance_profile(
    request: RootCriticalDistanceProfileRequest,
) -> RootCriticalDistanceProfile:
    return root_critical_distance_profile(
        request.polynomial,
        max_pair_rows=request.max_pair_rows,
    )


def compute_exact_splitting_field(
    request: ExactSplittingFieldRequest,
) -> ExactSplittingField:
    return exact_splitting_field(
        request.polynomial,
        embedding_index=request.embedding_index,
    )


def compute_splitting_field_distance_profile(
    request: SplittingFieldDistanceRequest,
) -> SplittingFieldDistanceProfile:
    return splitting_field_distance_profile(
        request.polynomial,
        request.splitting_field,
        max_pair_rows=request.max_pair_rows,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polynomial.root_critical_distance_profile.compute",
        title="Compute an exact root-critical distance profile",
        description=(
            "Return every distinct root of a bounded nonconstant univariate QQ "
            "polynomial, every distinct root of its derivative, their source "
            "multiplicities, and the complete Cartesian profile of exact squared "
            "complex distances. Root rectangles and real isolating intervals are "
            "certified rational enclosures; backend failures are operational "
            "errors and never imply an empty family."
        ),
        request_type=RootCriticalDistanceProfileRequest,
        result_type=RootCriticalDistanceProfile,
        run=compute_root_critical_distance_profile,
        tags=("polynomial", "roots", "critical-points", "distance", "exact"),
        examples=(
            OperationExample(
                name="cubic_roots_and_critical_point",
                description=(
                    "The three roots of z^3-1 are compared with the repeated "
                    "critical point 0; every squared distance is exactly 1. The "
                    "source must be a bounded nonconstant univariate polynomial "
                    "over QQ."
                ),
                input=_CUBIC_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.exact_splitting_field.compute",
        title="Compute the exact splitting field of the square-free p*p' support",
        description=(
            "Return the exact splitting field of the square-free support of "
            "one bounded nonconstant univariate QQ polynomial as one primitive "
            "element with its monic minimal polynomial, every distinct root as "
            "an exact rational polynomial in that element, and the exact "
            "conjugation automorphism. The degree is bounded and the blocking "
            "backend phases run in a killable child process."
        ),
        request_type=ExactSplittingFieldRequest,
        result_type=ExactSplittingField,
        run=compute_exact_splitting_field,
        tags=("polynomial", "splitting-field", "algebraic-number", "exact"),
        examples=(
            OperationExample(
                name="quadratic_splitting_field",
                description=(
                    "The splitting field of x^2-2 has defining polynomial t^2-2; "
                    "the source must be a bounded nonconstant univariate "
                    "polynomial over QQ."
                ),
                input={
                    "polynomial": {
                        "domain": "QQ",
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2],
                                },
                                {
                                    "coefficient": {"num": "-2", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    },
                    "embedding_index": 0,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.root_critical_distance_profile.splitting_field.compute",
        title="Bind the root-critical distance profile to an exact splitting field",
        description=(
            "Accept a supplied exact splitting field for the square-free "
            "support of p*p' and return the complete distinct root and critical "
            "families with multiplicities, the labelled Cartesian pair axis, "
            "every squared distance as an exact element of the field, and the "
            "certified nonnegative isolating interval. A field not bound to "
            "this exact source is rejected."
        ),
        request_type=SplittingFieldDistanceRequest,
        result_type=SplittingFieldDistanceProfile,
        run=compute_splitting_field_distance_profile,
        tags=(
            "polynomial",
            "roots",
            "critical-points",
            "distance",
            "splitting-field",
            "exact",
        ),
        examples=(
            OperationExample(
                name="double_root_distance",
                description=(
                    "x^2 has a double root at 0 and critical point 0, so the "
                    "bound distance is the exact zero of the trivial splitting "
                    "field; the field must be bound to this exact source."
                ),
                input=_DOUBLE_ROOT_EXAMPLE,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
