"""Declarations for rational polynomial multiplication."""

from jacobian.catalog._examples import example
from jacobian.math.polynomials._multiply_models import RationalPolynomialMultiplyRequest
from jacobian.math.polynomials._multiply_operations import rational_polynomial_multiply
from jacobian.math.polynomials._support import polynomial_operation
from jacobian.math.polynomials.values import RationalPolynomial

_C1 = {"num": "1", "den": "1"}

POLYNOMIAL_MULTIPLY_OPERATION = polynomial_operation(
    "polynomial.rational.multiply.compute",
    "Multiply rational polynomials",
    "Compute the exact product of two rational polynomials in the same QQ variable ring.",
    RationalPolynomialMultiplyRequest,
    RationalPolynomial,
    rational_polynomial_multiply,
    "polynomial",
    "rational",
    "multiplication",
    examples=(
        example(
            "multiply_x_plus_1_squared",
            "Multiply (x+1) * (x+1) = x^2+2x+1; both polynomials must use the same ordered variables.",
            {
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
