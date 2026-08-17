"""Domain-owned elementary integer and rational polynomial operations."""

from jacobian.catalog._examples import example
from jacobian.math.polynomial._elementary_operations import (
    integer_polynomial_compose,
    integer_polynomial_content,
    integer_polynomial_evaluate,
    integer_polynomial_gcd,
    integer_polynomial_primitive_part,
    integer_polynomial_shift,
    rational_partial_fraction_decomposition,
    rational_polynomial_derivative,
    rational_polynomial_division,
    rational_polynomial_evaluate,
    rational_polynomial_integral,
)
from jacobian.math.polynomial._models import (
    IntegerPolynomialCompositionRequest,
    IntegerPolynomialCompositionResult,
    IntegerPolynomialContentResult,
    IntegerPolynomialEvaluationRequest,
    IntegerPolynomialEvaluationResult,
    IntegerPolynomialGcdResult,
    IntegerPolynomialPairRequest,
    IntegerPolynomialPrimitivePartResult,
    IntegerPolynomialRequest,
    IntegerPolynomialShiftRequest,
    IntegerPolynomialShiftResult,
    RationalFunctionRequest,
    RationalPartialFractionResult,
    RationalPolynomialDerivativeResult,
    RationalPolynomialDivisionRequest,
    RationalPolynomialDivisionResult,
    RationalPolynomialEvaluationRequest,
    RationalPolynomialEvaluationResult,
    RationalPolynomialIntegralResult,
    RationalPolynomialRequest,
)
from jacobian.math.polynomial._support import polynomial_operation

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
        "polynomial.integer.compute.content",
        "Compute integer-polynomial content",
        "Compute the nonnegative coefficient GCD of one polynomial in ZZ[x].",
        IntegerPolynomialRequest,
        IntegerPolynomialContentResult,
        integer_polynomial_content,
        "polynomial",
        "integer",
        "content",
        examples=(
            example(
                "content_6x2_plus_9x",
                "Compute the coefficient content of 6x²+9x.",
                {"polynomial": {"coefficients": ["6", "9", "0"]}},
            ),
        ),
    ),
    polynomial_operation(
        "polynomial.integer.compute.primitive_part",
        "Compute an integer-polynomial primitive part",
        (
            "Separate one polynomial in ZZ[x] into nonnegative content and a "
            "primitive part, retaining an exact reconstruction."
        ),
        IntegerPolynomialRequest,
        IntegerPolynomialPrimitivePartResult,
        integer_polynomial_primitive_part,
        "polynomial",
        "integer",
        "primitive",
        examples=(
            example(
                "primitive_part_6x2_plus_9x",
                "Compute the primitive part of 6x²+9x.",
                {"polynomial": {"coefficients": ["6", "9", "0"]}},
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
        "polynomial.integer.compute.compose",
        "Compose integer polynomials",
        "Compute outer(inner(x)) exactly in ZZ[x] under a result-degree budget.",
        IntegerPolynomialCompositionRequest,
        IntegerPolynomialCompositionResult,
        integer_polynomial_compose,
        "polynomial",
        "integer",
        "composition",
        examples=(
            example(
                "compose_x_plus_one",
                "Compose x+1 with x².",
                {
                    "outer": {
                        "coefficient_order": "DESCENDING_DEGREE",
                        "coefficients": ["1", "1"],
                    },
                    "inner": {
                        "coefficient_order": "DESCENDING_DEGREE",
                        "coefficients": ["1", "0", "0"],
                    },
                },
            ),
        ),
    ),
    polynomial_operation(
        "polynomial.integer.compute.shift",
        "Shift an integer polynomial",
        "Compute p(x + a) exactly in ZZ[x].",
        IntegerPolynomialShiftRequest,
        IntegerPolynomialShiftResult,
        integer_polynomial_shift,
        "polynomial",
        "integer",
        "shift",
        examples=(
            example(
                "shift_x2_by_two",
                "Compute p(x+2) for p(x)=x².",
                {
                    "polynomial": {
                        "coefficient_order": "DESCENDING_DEGREE",
                        "coefficients": ["1", "0", "0"],
                    },
                    "shift": 2,
                },
            ),
        ),
    ),
)

RATIONAL_POLYNOMIAL_OPERATIONS = (
    polynomial_operation(
        "polynomial.rational.compute.quotient_remainder",
        "Divide rational polynomials",
        (
            "Compute quotient and remainder in QQ[x], retaining the exact "
            "dividend reconstruction."
        ),
        RationalPolynomialDivisionRequest,
        RationalPolynomialDivisionResult,
        rational_polynomial_division,
        "polynomial",
        "rational",
        "division",
        examples=(
            example(
                "divide_x_squared_minus_one",
                "Divide x²-1 by x-1.",
                {
                    "left": {
                        "polynomial_schema_version": "1",
                        "domain": "QQ",
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2],
                                },
                                {
                                    "coefficient": {"num": "-1", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    },
                    "right": {
                        "polynomial_schema_version": "1",
                        "domain": "QQ",
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [1],
                                },
                                {
                                    "coefficient": {"num": "-1", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    },
                },
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
                        "polynomial_schema_version": "1",
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
    polynomial_operation(
        "polynomial.rational.compute.derivative",
        "Differentiate a rational polynomial",
        "Compute the formal derivative of one bounded polynomial in QQ[x].",
        RationalPolynomialRequest,
        RationalPolynomialDerivativeResult,
        rational_polynomial_derivative,
        "polynomial",
        "rational",
        "derivative",
        examples=(
            example(
                "cubic_derivative",
                "Differentiate one half x³ minus 2x.",
                {
                    "polynomial": {
                        "polynomial_schema_version": "1",
                        "domain": "QQ",
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "2"},
                                    "exponents": [3],
                                },
                                {
                                    "coefficient": {"num": "-2", "den": "1"},
                                    "exponents": [1],
                                },
                            ]
                        },
                    }
                },
            ),
        ),
    ),
    polynomial_operation(
        "polynomial.rational.compute.integral",
        "Integrate a rational polynomial",
        (
            "Compute the unique formal antiderivative in QQ[x] whose "
            "integration constant is zero."
        ),
        RationalPolynomialRequest,
        RationalPolynomialIntegralResult,
        rational_polynomial_integral,
        "polynomial",
        "rational",
        "integration",
        examples=(
            example(
                "integral_two_x",
                "Integrate 2x with zero constant.",
                {
                    "polynomial": {
                        "polynomial_schema_version": "1",
                        "domain": "QQ",
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "2", "den": "1"},
                                    "exponents": [1],
                                }
                            ]
                        },
                    }
                },
            ),
        ),
    ),
    polynomial_operation(
        "polynomial.rational.compute.partial_fraction_decomposition",
        "Decompose a rational function over QQ",
        (
            "Compute a structured partial-fraction decomposition over QQ with "
            "monic denominator factors and an exact reduced reconstruction."
        ),
        RationalFunctionRequest,
        RationalPartialFractionResult,
        rational_partial_fraction_decomposition,
        "polynomial",
        "rational-function",
        "partial-fraction",
        examples=(
            example(
                "partial_fraction_one_over_x2_minus_one",
                "Decompose 1/(x²-1) into partial fractions.",
                {
                    "numerator": {
                        "polynomial_schema_version": "1",
                        "domain": "QQ",
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [0],
                                }
                            ]
                        },
                    },
                    "denominator": {
                        "polynomial_schema_version": "1",
                        "domain": "QQ",
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2],
                                },
                                {
                                    "coefficient": {"num": "-1", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    },
                },
            ),
        ),
    ),
)

__all__ = [
    "INTEGER_POLYNOMIAL_OPERATIONS",
    "RATIONAL_POLYNOMIAL_OPERATIONS",
]
