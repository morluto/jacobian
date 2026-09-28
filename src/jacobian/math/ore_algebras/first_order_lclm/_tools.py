"""Operation declaration for first-order differential LCLM."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.ore_algebras.first_order_lclm._models import (
    FirstOrderLCLMRequest,
    FirstOrderLCLMResult,
)
from jacobian.math.ore_algebras.first_order_lclm.operations import (
    differential_first_order_lclm,
)


def _coefficient(terms: list[tuple[int, int]]) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": ["x"],
        "numerator": {
            "terms": [
                {
                    "coefficient": {"num": str(value), "den": "1"},
                    "exponents": [degree],
                }
                for degree, value in sorted(terms, reverse=True)
                if value
            ]
        },
        "denominator": {
            "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [0]}]
        },
    }


def _operator(constant: int) -> dict[str, Any]:
    return {
        "variable": "x",
        "terms": [
            {"order": 0, "coefficient": _coefficient([(1, constant)])},
            {"order": 1, "coefficient": _coefficient([(0, 1)])},
        ],
    }


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="ore.differential.operator.first_order_lclm.compute",
        title="Compute the exact common left multiple of two first-order differential operators",
        description=(
            "Return the least common left multiple C and exact multipliers U,V "
            "such that C=U*L_left=V*L_right in QQ(x)<D>, where D a=aD+a'. "
            "Both inputs must have order one and bounded polynomial coefficients "
            "in QQ[x]. The output is not an analytic solution claim."
        ),
        request_type=FirstOrderLCLMRequest,
        result_type=FirstOrderLCLMResult,
        run=lambda request: differential_first_order_lclm(
            request.left, request.right
        ),
        tags=("ore-algebra", "differential-operator", "lclm", "exact"),
        discovery_terms=(
            "first-order differential operator least common left multiple",
            "common left multiple of Weyl algebra operators",
            "combine first-order differential annihilators",
        ),
        examples=(
            OperationExample(
                name="opposite_logarithmic_derivatives",
                description=(
                    "Compute a common left multiple of D-x and D+x; each input "
                    "must be a bounded polynomial-coefficient first-order operator."
                ),
                input={"left": _operator(-1), "right": _operator(1)},
            ),
        ),
    ),
)
