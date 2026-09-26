"""Manifest for exact monic differential right division."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.ore_algebras.differential_right_division._models import (
    DifferentialRightDivisionRequest,
    DifferentialRightDivisionResult,
)
from jacobian.math.ore_algebras.differential_right_division.operations import (
    differential_operator_right_divide_monic,
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
        operation_id="ore.differential.operator.right_division.compute",
        title="Right divide a differential Ore operator by a monic operator",
        description=(
            "Return the exact quotient Q and remainder R with A=Q*B+R and "
            "order(R)<order(B) in QQ(x)<D>, where D*a=a*D+a'. The bounded "
            "slice accepts integer-polynomial coefficients in ZZ[x] and a "
            "monic nonzero divisor, subject to the operation's exact work, "
            "coefficient-growth, and output bounds."
        ),
        request_type=DifferentialRightDivisionRequest,
        result_type=DifferentialRightDivisionResult,
        run=lambda request: differential_operator_right_divide_monic(
            request.dividend, request.divisor
        ),
        tags=("ore-algebra", "differential-operator", "right-division", "exact"),
        discovery_terms=(
            "right Euclidean division of differential operators",
            "quotient and remainder in QQ(x)<D>",
            "divide a differential operator on the right by a monic operator",
        ),
        examples=(
            OperationExample(
                name="monic_right_division_with_remainder",
                description=(
                    "Divide D^2+x by D+x using the convention A=Q*B+R; "
                    "the quotient is D-x and the remainder is x^2+x-1."
                ),
                input={
                    "dividend": {
                        "variable": "x",
                        "terms": [
                            {"order": 0, "coefficient": _coefficient([(1, 1)])},
                            {"order": 2, "coefficient": _coefficient([(0, 1)])},
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
