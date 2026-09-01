"""Declarations for rational polynomial multiplication."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials._multiply_kernel import rational_polynomial_multiply
from jacobian.math.polynomials._multiply_models import RationalPolynomialMultiplyRequest
from jacobian.math.polynomials.values import RationalPolynomial


def compute_rational_polynomial_multiply(
    request: RationalPolynomialMultiplyRequest,
) -> RationalPolynomial:
    return rational_polynomial_multiply(request.left, request.right)


_C1 = {"num": "1", "den": "1"}

POLYNOMIAL_MULTIPLY_OPERATION = MathTool(
    operation_id="polynomial.rational.multiply.compute",
    title="Multiply rational polynomials",
    description="Compute the exact product of two rational polynomials in the same QQ variable ring.",
    request_type=RationalPolynomialMultiplyRequest,
    result_type=RationalPolynomial,
    run=compute_rational_polynomial_multiply,
    tags=("polynomial", "rational", "multiplication"),
    examples=(
        OperationExample(
            name="multiply_x_plus_1_squared",
            description="Multiply (x+1) * (x+1) = x^2+2x+1; both polynomials must use the same ordered variables.",
            input={
                "left": {
                    "domain": "QQ",
                    "variables": ["x"],
                    "polynomial": {
                        "terms": [
                            {"coefficient": _C1, "exponents": [1]},
                            {"coefficient": _C1, "exponents": [0]},
                        ],
                    },
                },
                "right": {
                    "domain": "QQ",
                    "variables": ["x"],
                    "polynomial": {
                        "terms": [
                            {"coefficient": _C1, "exponents": [1]},
                            {"coefficient": _C1, "exponents": [0]},
                        ],
                    },
                },
            },
        ),
    ),
)

__all__ = ["POLYNOMIAL_MULTIPLY_OPERATION"]
