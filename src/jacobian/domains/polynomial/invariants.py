"""Exact polynomial invariant capabilities."""

from jacobian.contracts.polynomial_operations import (
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
from jacobian.domains._examples import example
from jacobian.domains.polynomial._support import polynomial_operation
from jacobian.domains.polynomial.operations import (
    polynomial_discriminant,
    polynomial_factorization,
    polynomial_gcd,
    polynomial_resultant,
    polynomial_square_free_decomposition,
)

POLYNOMIAL_INVARIANT_CAPABILITIES = (
    polynomial_operation(
        "polynomial.compute.gcd",
        "Compute a polynomial GCD and Bézout identity",
        "Compute the monic GCD of two bounded univariate polynomials over QQ.",
        PolynomialGcdRequest,
        PolynomialGcdResult,
        polynomial_gcd,
        "polynomial",
        "gcd",
        "bezout",
        invocation_examples=(
            example(
                "gcd_x2_minus_one_x_minus_one",
                "Compute the GCD of x²-1 and x-1.",
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
        "polynomial.compute.resultant",
        "Compute a polynomial resultant",
        "Compute the exact resultant in one named elimination variable over QQ.",
        PolynomialResultantRequest,
        PolynomialResultantResult,
        polynomial_resultant,
        "polynomial",
        "resultant",
        "elimination",
        invocation_examples=(
            example(
                "resultant_x2_minus_one_x_minus_two",
                "Compute the resultant of x²-1 and x-2.",
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
        polynomial_discriminant,
        "polynomial",
        "discriminant",
        invocation_examples=(
            example(
                "discriminant_x2_minus_one",
                "Compute the discriminant of x²-1.",
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
        polynomial_square_free_decomposition,
        "polynomial",
        "square-free",
        "multiplicity",
        invocation_examples=(
            example(
                "square_free_x2_minus_one",
                "Compute the square-free decomposition of x²-1.",
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
        polynomial_factorization,
        "polynomial",
        "factorization",
        "exact-computation",
        invocation_examples=(
            example(
                "factor_x_squared_minus_one",
                "Factor x²-1 over QQ.",
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

__all__ = ["POLYNOMIAL_INVARIANT_CAPABILITIES"]
