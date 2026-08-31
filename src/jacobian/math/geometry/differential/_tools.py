"""Rational coordinate-tensor operation declaration."""

from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.geometry.differential._models import (
    RationalLieDerivativeProfile,
    RationalLieDerivativeRequest,
)
from jacobian.math.geometry.differential.operations import lie_derivative


def _compute(request: RationalLieDerivativeRequest) -> RationalLieDerivativeProfile:
    return lie_derivative(request.vector_field, request.tensor)


def _polynomial(
    variables: tuple[str, ...], *terms: tuple[int, tuple[int, ...]]
) -> dict[str, Any]:
    return {
        "variables": list(variables),
        "numerator": {
            "terms": [
                {
                    "coefficient": {"num": str(coefficient), "den": "1"},
                    "exponents": list(exponents),
                }
                for coefficient, exponents in terms
            ]
        },
        "denominator": {
            "terms": [
                {
                    "coefficient": {"num": "1", "den": "1"},
                    "exponents": [0] * len(variables),
                }
            ]
        },
    }


TOOLS = (
    MathTool(
        operation_id="differential_geometry.rational_tensor.lie_derivative.compute",
        title="Compute the Lie derivative of a rational coordinate tensor",
        description=(
            "Return the complete exact mixed coordinate tensor L_X T over "
            "QQ(x_1,...,x_n). The supplied rank-one contravariant vector field "
            "and tensor must share one ordered axis. Components use typed "
            "sparse rational functions; their inherited nonvanishing guards "
            "are retained even when canonical normalization cancels a pole."
        ),
        request_type=RationalLieDerivativeRequest,
        result_type=RationalLieDerivativeProfile,
        run=_compute,
        tags=(
            "differential-geometry",
            "Lie-derivative",
            "tensor",
            "rational-function",
            "exact",
        ),
        discovery_terms=("coordinate tensor", "Lie bracket", "infinitesimal pullback"),
        examples=(
            example(
                "scalar_directional_derivative",
                "Compute L_X(x^2 y) for X = y partial_x + x partial_y on "
                "the ordered chart (x, y). Rank-zero tensors have one component.",
                {
                    "vector_field": {
                        "coordinate_axis": ["x", "y"],
                        "variance": ["CONTRAVARIANT"],
                        "components": [
                            _polynomial(("x", "y"), (1, (0, 1))),
                            _polynomial(("x", "y"), (1, (1, 0))),
                        ],
                        "retained_nonzero_denominators": [],
                    },
                    "tensor": {
                        "coordinate_axis": ["x", "y"],
                        "variance": [],
                        "components": [_polynomial(("x", "y"), (1, (2, 1)))],
                        "retained_nonzero_denominators": [],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
