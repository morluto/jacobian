"""Domain-owned elementary integer and rational polynomial operations."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials._elementary_kernel import (
    integer_polynomial_evaluate,
    integer_polynomial_gcd,
    rational_polynomial_derivative,
    rational_polynomial_evaluate,
)
from jacobian.math.polynomials._models import (
    IntegerPolynomialEvaluationRequest,
    IntegerPolynomialEvaluationResult,
    IntegerPolynomialGcdResult,
    IntegerPolynomialPairRequest,
    RationalPolynomialDerivativeResult,
    RationalPolynomialEvaluationRequest,
    RationalPolynomialEvaluationResult,
    RationalPolynomialRequest,
)


def _run_integer_gcd(
    request: IntegerPolynomialPairRequest,
) -> IntegerPolynomialGcdResult:
    return integer_polynomial_gcd(request.left, request.right)


def _run_integer_evaluation(
    request: IntegerPolynomialEvaluationRequest,
) -> IntegerPolynomialEvaluationResult:
    return integer_polynomial_evaluate(request.polynomial, request.point)


def _run_rational_evaluation(
    request: RationalPolynomialEvaluationRequest,
) -> RationalPolynomialEvaluationResult:
    return rational_polynomial_evaluate(request.polynomial, request.point)


def _run_rational_derivative(
    request: RationalPolynomialRequest,
) -> RationalPolynomialDerivativeResult:
    return rational_polynomial_derivative(request.polynomial)


INTEGER_POLYNOMIAL_OPERATIONS = (
    MathTool(
        operation_id="polynomial.integer.compute.gcd",
        title="Compute an integer-polynomial GCD",
        description=(
            "Compute the nonnegative-leading GCD in ZZ[x], including the content "
            "of both inputs and the result."
        ),
        request_type=IntegerPolynomialPairRequest,
        result_type=IntegerPolynomialGcdResult,
        run=_run_integer_gcd,
        tags=("polynomial", "integer", "gcd"),
        examples=(
            OperationExample(
                name="integer_gcd",
                description="Compute the GCD of two integer polynomials.",
                input={
                    "left": {"coefficients": ["6", "6", "0"]},
                    "right": {"coefficients": ["8", "8", "0"]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.integer.compute.evaluate",
        title="Evaluate an integer polynomial",
        description="Evaluate one bounded polynomial in ZZ[x] at an exact integer point.",
        request_type=IntegerPolynomialEvaluationRequest,
        result_type=IntegerPolynomialEvaluationResult,
        run=_run_integer_evaluation,
        tags=("polynomial", "integer", "evaluation"),
        examples=(
            OperationExample(
                name="evaluate_at_four",
                description="Evaluate 2x²-3x+1 at 4.",
                input={"polynomial": {"coefficients": ["2", "-3", "1"]}, "point": "4"},
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.rational.compute.evaluate",
        title="Evaluate a rational polynomial",
        description="Evaluate one bounded polynomial in QQ[x] at an exact rational point.",
        request_type=RationalPolynomialEvaluationRequest,
        result_type=RationalPolynomialEvaluationResult,
        run=_run_rational_evaluation,
        tags=("polynomial", "rational", "evaluation"),
        examples=(
            OperationExample(
                name="rational_evaluate_x2_plus_one",
                description="Evaluate x²+1 at 2.",
                input={
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
    MathTool(
        operation_id="polynomial.rational.compute.derivative",
        title="Differentiate a rational polynomial",
        description="Compute the formal derivative of one bounded polynomial in QQ[x].",
        request_type=RationalPolynomialRequest,
        result_type=RationalPolynomialDerivativeResult,
        run=_run_rational_derivative,
        tags=("polynomial", "rational", "derivative"),
        examples=(
            OperationExample(
                name="cubic_derivative",
                description="Differentiate one half x³ minus 2x.",
                input={
                    "polynomial": {
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
)

__all__ = ["INTEGER_POLYNOMIAL_OPERATIONS"]
