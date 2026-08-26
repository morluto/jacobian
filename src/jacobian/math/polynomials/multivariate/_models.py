"""Shared bounds and helpers for exact multivariate polynomial contracts."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from jacobian.math.polynomials.values import RationalPolynomial


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("polynomial.multivariate_contract", message)


_MAX_MULTIVARIATE_TERMS = 512
_MAX_MULTIVARIATE_EXPONENT = 64
_MAX_MULTIVARIATE_COEFFICIENT_DIGITS = 256
_MAX_ELIMINATION_DEGREE_SUM = 64
_MULTIVARIATE_MIN_VARIABLES = 2


def _validate_multivariate_pair(
    left: RationalPolynomial,
    right: RationalPolynomial,
) -> None:
    """Shared validation for two polynomials in the same declared ring."""

    if len(left.variables) < _MULTIVARIATE_MIN_VARIABLES:
        raise _validation_error(
            "multivariate operations require at least two variables"
        )
    if left.variables != right.variables:
        raise _validation_error("both polynomials must use the same ordered variables")


def _degree_in_variable(polynomial: RationalPolynomial, variable_index: int) -> int:
    return max(
        (term.exponents[variable_index] for term in polynomial.polynomial.terms),
        default=0,
    )


def _remaining_total_degree(
    polynomial: RationalPolynomial,
    variable_index: int,
) -> int:
    return max(
        (
            sum(
                exponent
                for index, exponent in enumerate(term.exponents)
                if index != variable_index
            )
            for term in polynomial.polynomial.terms
        ),
        default=0,
    )


def _maximum_coefficient_support(
    polynomial: RationalPolynomial,
    variable_index: int,
) -> int:
    support_by_degree: dict[int, int] = {}
    for term in polynomial.polynomial.terms:
        degree = term.exponents[variable_index]
        support_by_degree[degree] = support_by_degree.get(degree, 0) + 1
    return max(support_by_degree.values(), default=0)
