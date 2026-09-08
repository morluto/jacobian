"""The scalar rational-function field gradient declaration."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.rational_functions.gradient._models import (
    RationalFunctionGradient,
    RationalGradientRequest,
)
from jacobian.math.polynomials.rational_functions.gradient.operations import gradient


def _compute(request: RationalGradientRequest) -> RationalFunctionGradient:
    return gradient(request.function)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="rational_function.gradient.compute",
        title="Differentiate a rational function in every coordinate",
        description=(
            "Return the complete exact partial derivatives of a canonical "
            "multivariate QQ rational function on its ordered coordinate axis. "
            "Normalize quotient-rule derivatives in the same rational-function "
            "field. Constants on empty axes have empty gradients; zero "
            "components retain all declared coordinates."
        ),
        request_type=RationalGradientRequest,
        result_type=RationalFunctionGradient,
        run=_compute,
        tags=("polynomial", "rational-function", "gradient", "derivative", "exact"),
        examples=(
            OperationExample(
                name="two_variable_quotient",
                description="Differentiate (x²+y)/(x-y) on the declared (x,y) axis.",
                input={
                    "function": {
                        "variables": ["x", "y"],
                        "numerator": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2, 0],
                                },
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [0, 1],
                                },
                            ]
                        },
                        "denominator": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [1, 0],
                                },
                                {
                                    "coefficient": {"num": "-1", "den": "1"},
                                    "exponents": [0, 1],
                                },
                            ]
                        },
                    }
                },
            ),
        ),
    ),
)
