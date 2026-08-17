"""Contracts for exact multivariate polynomial operations over ``QQ``."""

from __future__ import annotations

from math import comb
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)

_MAX_MULTIVARIATE_TERMS = 512
_MAX_MULTIVARIATE_EXPONENT = 64
_MAX_MULTIVARIATE_COEFFICIENT_DIGITS = 256
_MAX_ELIMINATION_DEGREE_SUM = 64
# The result converter rejects sparse outputs above this size.  The request
# validator uses the same bound to reject large possible supports before
# SymPy expands the Sylvester determinant.
_MAX_RESULTANT_TERMS = 1_024

MonomialOrder = Literal["lex", "grlex", "grevlex"]
"""Declared monomial order for multivariate polynomial division."""

_MULTIVARIATE_MIN_VARIABLES = 2


def _validate_multivariate_pair(
    left: RationalPolynomial,
    right: RationalPolynomial,
) -> None:
    """Shared validation for two polynomials in the same declared ring."""

    if len(left.variables) < _MULTIVARIATE_MIN_VARIABLES:
        raise ValueError("multivariate operations require at least two variables")
    if left.variables != right.variables:
        raise ValueError("both polynomials must use the same ordered variables")


def _resultant_support_bound(
    left: RationalPolynomial,
    right: RationalPolynomial,
    elimination_index: int,
) -> int:
    """Bound the number of possible monomials in a multivariate resultant.

    If ``f`` and ``g`` have elimination degrees ``m`` and ``n``, and total
    remaining-variable degrees ``d_f`` and ``d_g``, every resultant monomial
    has total degree at most ``n*d_f + m*d_g``.  The returned binomial is the
    number of monomials up to that degree in the remaining variables.
    """

    remaining_variable_count = len(left.variables) - 1
    if remaining_variable_count == 0:
        return 1

    def degree(polynomial: RationalPolynomial, *, in_remaining: bool) -> int:
        return max(
            (
                sum(
                    exponent
                    for index, exponent in enumerate(term.exponents)
                    if (index != elimination_index) == in_remaining
                )
                for term in polynomial.polynomial.terms
            ),
            default=0,
        )

    left_elimination_degree = degree(left, in_remaining=False)
    right_elimination_degree = degree(right, in_remaining=False)
    left_remaining_degree = degree(left, in_remaining=True)
    right_remaining_degree = degree(right, in_remaining=True)
    resultant_degree_bound = (
        right_elimination_degree * left_remaining_degree
        + left_elimination_degree * right_remaining_degree
    )
    return comb(
        resultant_degree_bound + remaining_variable_count,
        remaining_variable_count,
    )


class MultivariateGcdRequest(StrictModel):
    """Two multivariate polynomials in ``QQ[x_1, ..., x_n]``."""

    left: RationalPolynomial
    right: RationalPolynomial

    @model_validator(mode="after")
    def require_multivariate_ring(self) -> Self:
        _validate_multivariate_pair(self.left, self.right)
        for polynomial in (self.left, self.right):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_MULTIVARIATE_TERMS,
                maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
                maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
            )
        return self


class MultivariateGcdResult(StrictModel):
    gcd: RationalPolynomial
    convention: Literal["MONIC_ASSOCIATE"] = "MONIC_ASSOCIATE"


class MultivariateDivisionRequest(StrictModel):
    """Divide one multivariate polynomial by another under a declared monomial order."""

    left: RationalPolynomial
    right: RationalPolynomial
    monomial_order: MonomialOrder = "lex"

    @model_validator(mode="after")
    def require_multivariate_ring(self) -> Self:
        _validate_multivariate_pair(self.left, self.right)
        if not self.right.polynomial.terms:
            raise ValueError("divisor polynomial must be nonzero")
        for polynomial in (self.left, self.right):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_MULTIVARIATE_TERMS,
                maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
                maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
            )
        return self


class MultivariateDivisionResult(StrictModel):
    quotient: RationalPolynomial
    remainder: RationalPolynomial
    monomial_order: MonomialOrder
    convention: Literal["EXACT_DIVISION_REMAINDER"] = "EXACT_DIVISION_REMAINDER"


class MultivariateResultantRequest(StrictModel):
    """Compute a bounded resultant with respect to one variable.

    The request rejects inputs whose degree envelope can produce more terms
    than the exact sparse result contract can represent.
    """

    left: RationalPolynomial
    right: RationalPolynomial
    elimination_variable: PolynomialVariable = Field(
        description="Variable eliminated by the Sylvester resultant.",
    )

    @model_validator(mode="after")
    def require_multivariate_ring(self) -> Self:
        _validate_multivariate_pair(self.left, self.right)
        if self.elimination_variable not in self.left.variables:
            raise ValueError("elimination variable must belong to the declared ring")
        for polynomial in (self.left, self.right):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_MULTIVARIATE_TERMS,
                maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
                maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
            )
        variable_index = self.left.variables.index(self.elimination_variable)
        for polynomial, label in ((self.left, "left"), (self.right, "right")):
            degree_in_variable = max(
                (
                    term.exponents[variable_index]
                    for term in polynomial.polynomial.terms
                ),
                default=0,
            )
            if degree_in_variable == 0:
                raise ValueError(
                    f"{label} polynomial has zero degree in the elimination variable"
                )
        degree_sum = max(
            (term.exponents[variable_index] for term in self.left.polynomial.terms),
            default=0,
        ) + max(
            (term.exponents[variable_index] for term in self.right.polynomial.terms),
            default=0,
        )
        if degree_sum > _MAX_ELIMINATION_DEGREE_SUM:
            raise ValueError("Sylvester degree exceeds the resultant budget")
        if (
            _resultant_support_bound(self.left, self.right, variable_index)
            > _MAX_RESULTANT_TERMS
        ):
            raise ValueError("resultant output exceeds the term budget")
        return self


class MultivariateScalarValue(StrictModel):
    kind: Literal["SCALAR"] = "SCALAR"
    value: CanonicalRational


class MultivariatePolynomialValue(StrictModel):
    kind: Literal["POLYNOMIAL"] = "POLYNOMIAL"
    value: RationalPolynomial


MultivariateInvariantValue = Annotated[
    MultivariateScalarValue | MultivariatePolynomialValue,
    Field(discriminator="kind"),
]


class MultivariateResultantResult(StrictModel):
    elimination_variable: PolynomialVariable
    resultant: MultivariateInvariantValue
    convention: Literal["SYLVESTER_DETERMINANT"] = "SYLVESTER_DETERMINANT"


__all__ = [
    "MonomialOrder",
    "MultivariateDivisionRequest",
    "MultivariateDivisionResult",
    "MultivariateGcdRequest",
    "MultivariateGcdResult",
    "MultivariateInvariantValue",
    "MultivariatePolynomialValue",
    "MultivariateResultantRequest",
    "MultivariateResultantResult",
    "MultivariateScalarValue",
]
