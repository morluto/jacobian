"""Selected-variable discrete antiderivative declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials._discrete_antiderivative import (
    RationalDiscreteAntiderivativeRequest,
    RationalDiscreteAntiderivativeResult,
    rational_discrete_antiderivative,
)

RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION = MathTool(
    operation_id="polynomial.rational.discrete_antiderivative.compute",
    title="Compute a selected-variable rational discrete antiderivative",
    description="Return the unique normalized Q with Q at the selected variable zero equal to zero and Q(x+1)-Q(x)=P, retaining other variables as coefficient parameters.",
    request_type=RationalDiscreteAntiderivativeRequest,
    result_type=RationalDiscreteAntiderivativeResult,
    run=rational_discrete_antiderivative,
    tags=("polynomial", "finite-difference", "antiderivative", "rational", "exact"),
    examples=(
        OperationExample(
            name="square",
            description=(
                "Compute the zero-based discrete antiderivative of k squared "
                "with respect to k; the selected variable must be one of the "
                "polynomial's declared axes."
            ),
            input={
                "polynomial": {
                    "variables": ["k"],
                    "polynomial": {
                        "terms": [
                            {"coefficient": {"num": "1", "den": "1"}, "exponents": [2]}
                        ]
                    },
                },
                "variable": "k",
            },
        ),
    ),
)

__all__ = ["RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION"]
