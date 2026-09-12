"""Selected-variable discrete antiderivative declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials._discrete_antiderivative import (
    RationalDiscreteAntiderivativeRequest,
    RationalDiscreteAntiderivativeResult,
    _compute_discrete_antiderivative,
)


def _run(
    request: RationalDiscreteAntiderivativeRequest,
) -> RationalDiscreteAntiderivativeResult:
    return _compute_discrete_antiderivative(request.polynomial, request.variable)


RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION = MathTool(
    operation_id="polynomial.rational.discrete_antiderivative.compute",
    title="Compute a selected-variable rational discrete antiderivative",
    description=(
        "Return the unique normalized Q with Q at the selected variable zero "
        "equal to zero and Q(x+1)-Q(x)=P, retaining other variables as "
        "coefficient parameters. The current kernel admits at most 1000000 "
        "exact rational updates; larger triangular work is a resource-admission "
        "failure."
    ),
    request_type=RationalDiscreteAntiderivativeRequest,
    result_type=RationalDiscreteAntiderivativeResult,
    run=_run,
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
