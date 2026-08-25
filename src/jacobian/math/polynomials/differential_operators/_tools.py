"""Differential-operator operation declarations."""

from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.polynomials.differential_operators._models import (
    DifferentialOperatorApplyRequest,
    DifferentialOperatorApplyResult,
)
from jacobian.math.polynomials.differential_operators._operations import (
    compute_differential_operator_application,
)


def _polynomial(
    variables: tuple[str, ...],
    *terms: tuple[tuple[int, int], tuple[int, ...]],
) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": list(variables),
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": str(numerator), "den": str(denominator)},
                    "exponents": list(exponents),
                }
                for (numerator, denominator), exponents in terms
            ]
        },
    }


VARIABLES = ("x", "y")

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polynomial.differential_operator.apply.compute",
        title="Apply a constant-coefficient differential operator",
        description=(
            "Return D^k(f) exactly for a canonical sparse polynomial f over QQ and "
            "a finite constant-coefficient multi-index operator D on the same ordered "
            "variable axis. The result retains the finite source relation, reports "
            "zero and optional expected-polynomial equality, and makes no conclusion "
            "about other iteration counts."
        ),
        request_type=DifferentialOperatorApplyRequest,
        result_type=DifferentialOperatorApplyResult,
        run=compute_differential_operator_application,
        tags=(
            "polynomial",
            "multivariate",
            "differential-operator",
            "partial-derivative",
            "constant-coefficient",
            "exact",
        ),
        examples=(
            example(
                "second_iterate_of_dx_minus_dy",
                "Apply (partial_x - partial_y)^2 to x^2*y + 3*y^2 and compare "
                "with -4*x + 2*y + 6; polynomial and operator must carry the "
                "same complete ordered axis and canonical term order.",
                {
                    "polynomial": _polynomial(
                        VARIABLES,
                        ((1, 1), (2, 1)),
                        ((3, 1), (0, 2)),
                    ),
                    "operator": {
                        "domain": "QQ",
                        "variables": list(VARIABLES),
                        "terms": [
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "orders": [1, 0],
                            },
                            {
                                "coefficient": {"num": "-1", "den": "1"},
                                "orders": [0, 1],
                            },
                        ],
                    },
                    "iterations": 2,
                    "expected": _polynomial(
                        VARIABLES,
                        ((-4, 1), (1, 0)),
                        ((2, 1), (0, 1)),
                        ((6, 1), (0, 0)),
                    ),
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
