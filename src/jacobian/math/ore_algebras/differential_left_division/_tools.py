"""Manifest for exact monic differential left division."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.ore_algebras.differential_left_division._models import (
    DifferentialLeftDivisionRequest,
    DifferentialLeftDivisionResult,
)
from jacobian.math.ore_algebras.differential_left_division.operations import (
    differential_operator_left_divide_monic,
)


def _coefficient(terms: list[tuple[int, int]]) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": ["x"],
        "numerator": {
            "terms": [
                {
                    "coefficient": {"num": value, "den": 1},
                    "exponents": [degree],
                }
                for degree, value in sorted(terms, reverse=True)
                if value
            ]
        },
        "denominator": {
            "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [0]}]
        },
    }


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="ore.differential.operator.left_division.compute",
        title="Left divide a differential Ore operator by a monic operator",
        description=(
            "Return the exact quotient Q and remainder R with A=B*Q+R and "
            "order(R)<order(B) in QQ(x)<D>, where D*a=a*D+a'. The bounded "
            "slice accepts order-at-most-four operators with integer polynomial "
            "coefficients in ZZ[x] and a monic nonzero divisor."
        ),
        request_type=DifferentialLeftDivisionRequest,
        result_type=DifferentialLeftDivisionResult,
        run=lambda request: differential_operator_left_divide_monic(
            request.dividend, request.divisor
        ),
        tags=("ore-algebra", "differential-operator", "left-division", "exact"),
        discovery_terms=(
            "left Euclidean division of differential operators",
            "quotient and remainder in QQ(x)<D>",
            "divide a differential operator on the left by a monic operator",
        ),
        examples=(
            OperationExample(
                name="monic_left_division_with_nonconstant_quotient",
                description=(
                    "Divide x*D^2+(x^2+2)*D+x+1 by D+x using A=B*Q+R; "
                    "the quotient is x*D+1 and the remainder is one."
                ),
                input={
                    "dividend": {
                        "variable": "x",
                        "terms": [
                            {"order": 0, "coefficient": _coefficient([(1, 1), (0, 1)])},
                            {"order": 1, "coefficient": _coefficient([(2, 1), (0, 2)])},
                            {"order": 2, "coefficient": _coefficient([(1, 1)])},
                        ],
                    },
                    "divisor": {
                        "variable": "x",
                        "terms": [
                            {"order": 0, "coefficient": _coefficient([(1, 1)])},
                            {"order": 1, "coefficient": _coefficient([(0, 1)])},
                        ],
                    },
                },
            ),
        ),
    ),
)
