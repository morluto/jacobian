"""Domain-owned elementary integer and rational polynomial operations."""

from jacobian.catalog._examples import example
from jacobian.math.polynomials._elementary_operations import (
    integer_polynomial_evaluate,
    integer_polynomial_gcd,
    rational_polynomial_evaluate,
)
from jacobian.math.polynomials._models import (
    IntegerPolynomialEvaluationRequest,
    IntegerPolynomialEvaluationResult,
    IntegerPolynomialGcdResult,
    IntegerPolynomialPairRequest,
    RationalPolynomialEvaluationRequest,
    RationalPolynomialEvaluationResult,
)
from jacobian.math.polynomials._support import polynomial_operation

INTEGER_POLYNOMIAL_OPERATIONS = (
    polynomial_operation(
        "polynomial.integer.compute.gcd",
        "Compute an integer-polynomial GCD",
        (
            "Compute the nonnegative-leading GCD in ZZ[x], including the content "
            "of both inputs and the result."
        ),
        IntegerPolynomialPairRequest,
        IntegerPolynomialGcdResult,
        integer_polynomial_gcd,
        "polynomial",
        "integer",
        "gcd",
        examples=(
            example(
                "integer_gcd",
                "Compute the GCD of two integer polynomials.",
                {
                    "left": {"coefficients": ["6", "6", "0"]},
                    "right": {"coefficients": ["8", "8", "0"]},
                },
            ),
        ),
    ),
    polynomial_operation(
        "polynomial.integer.compute.evaluate",
        "Evaluate an integer polynomial",
        "Evaluate one bounded polynomial in ZZ[x] at an exact integer point.",
        IntegerPolynomialEvaluationRequest,
        IntegerPolynomialEvaluationResult,
        integer_polynomial_evaluate,
        "polynomial",
        "integer",
        "evaluation",
        examples=(
            example(
                "evaluate_at_four",
                "Evaluate 2x²-3x+1 at 4.",
                {"polynomial": {"coefficients": ["2", "-3", "1"]}, "point": "4"},
            ),
        ),
    ),
    polynomial_operation(
        "polynomial.rational.compute.evaluate",
        "Evaluate a rational polynomial",
        "Evaluate one bounded polynomial in QQ[x] at an exact rational point.",
        RationalPolynomialEvaluationRequest,
        RationalPolynomialEvaluationResult,
        rational_polynomial_evaluate,
        "polynomial",
        "rational",
        "evaluation",
        examples=(
            example(
                "rational_evaluate_x2_plus_one",
                "Evaluate x²+1 at 2.",
                {
                    "polynomial": {
                        "domain": "QQ",
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2],
                                },
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    },
                    "point": {"num": "2", "den": "1"},
                },
            ),
        ),
    ),
)

__all__ = ["INTEGER_POLYNOMIAL_OPERATIONS"]
