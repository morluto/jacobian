"""Exact polynomial invariant operations."""

from jacobian.catalog._examples import example
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
from jacobian.math.polynomials._support import polynomial_operation
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
    polynomial_operation(
        "polynomial.compute.gcd",
        "Compute a polynomial GCD and Bézout identity",
        "Compute the monic GCD of two bounded univariate polynomials over QQ.",
        PolynomialGcdRequest,
        PolynomialGcdResult,
        _run_gcd,
        "polynomial",
        "gcd",
        "bezout",
        examples=(
            example(
                "gcd_x2_minus_one_x_minus_one",
                "Compute the GCD of x²-1 and x-1.",
                {
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
    polynomial_operation(
        "polynomial.compute.resultant",
        "Compute a polynomial resultant",
        "Compute the exact resultant in one named elimination variable over QQ.",
        PolynomialResultantRequest,
        PolynomialResultantResult,
        _run_resultant,
        "polynomial",
        "resultant",
        "elimination",
        "univariate",
        "rational",
        examples=(
            example(
                "resultant_x2_minus_one_x_minus_two",
                "Compute the resultant of x²-1 and x-2.",
                {
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
    polynomial_operation(
        "polynomial.compute.discriminant",
        "Compute a polynomial discriminant",
        "Compute the standard exact discriminant in one named variable over QQ.",
        PolynomialDiscriminantRequest,
        PolynomialDiscriminantResult,
        _run_discriminant,
        "polynomial",
        "discriminant",
        examples=(
            example(
                "discriminant_x2_minus_one",
                "Compute the discriminant of x²-1.",
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
    polynomial_operation(
        "polynomial.compute.square_free_decomposition",
        "Compute a square-free decomposition",
        "Decompose a bounded polynomial over QQ into monic square-free factors.",
        PolynomialSquareFreeRequest,
        PolynomialSquareFreeDecompositionResult,
        _run_square_free,
        "polynomial",
        "square-free",
        "multiplicity",
        examples=(
            example(
                "square_free_x2_minus_one",
                "Compute the square-free decomposition of x²-1.",
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
    polynomial_operation(
        "polynomial.factor.compute",
        "Factor a univariate rational polynomial",
        (
            "Compute a rational content and multiplicity-bearing monic irreducible "
            "factors over QQ, together with an exact reconstructed product. Factor "
            "irreducibility is not independently certified by this producer."
        ),
        PolynomialFactorRequest,
        PolynomialFactorizationResult,
        _run_factorization,
        "polynomial",
        "factorization",
        "exact-computation",
        examples=(
            example(
                "factor_x_squared_minus_one",
                "Factor x²-1 over QQ.",
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
