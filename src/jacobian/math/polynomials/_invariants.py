"""Exact polynomial invariant operations."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials._models import (
    PolynomialDiscriminantRequest,
    PolynomialDiscriminantResult,
    PolynomialFactorizationResult,
    PolynomialFactorRequest,
    PolynomialGcdRequest,
    PolynomialGcdResult,
    PolynomialResultantRequest,
    PolynomialResultantResult,
    PolynomialSquareFreeDecompositionResult,
    PolynomialSquareFreeRequest,
)
from jacobian.math.polynomials.operations import (
    polynomial_discriminant,
    polynomial_factorization,
    polynomial_gcd,
    polynomial_resultant,
    polynomial_square_free_decomposition,
)


def _run_gcd(request: PolynomialGcdRequest) -> PolynomialGcdResult:
    return polynomial_gcd(request.left, request.right)


def _run_resultant(request: PolynomialResultantRequest) -> PolynomialResultantResult:
    return polynomial_resultant(
        request.left, request.right, request.elimination_variable
    )


def _run_discriminant(
    request: PolynomialDiscriminantRequest,
) -> PolynomialDiscriminantResult:
    return polynomial_discriminant(request.polynomial, request.variable)


def _run_square_free(
    request: PolynomialSquareFreeRequest,
) -> PolynomialSquareFreeDecompositionResult:
    return polynomial_square_free_decomposition(request.polynomial)


def _run_factorization(
    request: PolynomialFactorRequest,
) -> PolynomialFactorizationResult:
    return polynomial_factorization(request.polynomial)


POLYNOMIAL_INVARIANT_OPERATIONS = (
    MathTool(
        operation_id="polynomial.compute.gcd",
        title="Compute a polynomial GCD and Bézout identity",
        description="Compute the monic GCD of two bounded univariate polynomials over QQ.",
        request_type=PolynomialGcdRequest,
        result_type=PolynomialGcdResult,
        run=_run_gcd,
        tags=("polynomial", "gcd", "bezout"),
        examples=(
            OperationExample(
                name="gcd_x2_minus_one_x_minus_one",
                description="Compute the GCD of x²-1 and x-1.",
                input={
                    "left": {
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
    MathTool(
        operation_id="polynomial.compute.resultant",
        title="Compute a polynomial resultant",
        description="Compute the exact resultant in one named elimination variable over QQ.",
        request_type=PolynomialResultantRequest,
        result_type=PolynomialResultantResult,
        run=_run_resultant,
        tags=("polynomial", "resultant", "elimination", "univariate", "rational"),
        examples=(
            OperationExample(
                name="resultant_x2_minus_one_x_minus_two",
                description="Compute the resultant of x²-1 and x-2.",
                input={
                    "left": {
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
                        "domain": "QQ",
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [1],
                                },
                                {
                                    "coefficient": {"num": "-2", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    },
                    "elimination_variable": "x",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.compute.discriminant",
        title="Compute a polynomial discriminant",
        description="Compute the standard exact discriminant in one named variable over QQ.",
        request_type=PolynomialDiscriminantRequest,
        result_type=PolynomialDiscriminantResult,
        run=_run_discriminant,
        tags=("polynomial", "discriminant"),
        examples=(
            OperationExample(
                name="discriminant_x2_minus_one",
                description="Compute the discriminant of x²-1.",
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
                                    "coefficient": {"num": "-1", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    },
                    "variable": "x",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.compute.square_free_decomposition",
        title="Compute a square-free decomposition",
        description="Decompose a bounded polynomial over QQ into monic square-free factors.",
        request_type=PolynomialSquareFreeRequest,
        result_type=PolynomialSquareFreeDecompositionResult,
        run=_run_square_free,
        tags=("polynomial", "square-free", "multiplicity"),
        examples=(
            OperationExample(
                name="square_free_x2_minus_one",
                description="Compute the square-free decomposition of x²-1.",
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
                                    "coefficient": {"num": "-1", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.factor.compute",
        title="Factor a univariate rational polynomial",
        description=(
            "Compute a rational content and multiplicity-bearing monic irreducible "
            "factors over QQ, together with an exact reconstructed product. Factor "
            "irreducibility is not independently certified by this producer."
        ),
        request_type=PolynomialFactorRequest,
        result_type=PolynomialFactorizationResult,
        run=_run_factorization,
        tags=("polynomial", "factorization", "exact-computation"),
        examples=(
            OperationExample(
                name="factor_x_squared_minus_one",
                description="Factor x²-1 over QQ.",
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
                                    "coefficient": {"num": "-1", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    }
                },
            ),
        ),
    ),
)

__all__ = ["POLYNOMIAL_INVARIANT_OPERATIONS"]
