"""Contracts for exact multivariate polynomial operations over ``QQ``."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)

_MAX_MULTIVARIATE_TERMS = 512
_MAX_MULTIVARIATE_EXPONENT = 64
_MAX_MULTIVARIATE_COEFFICIENT_DIGITS = 256
_MAX_ELIMINATION_DEGREE_SUM = 64

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


class MultivariateGcdRequest(ContractModel):
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


class MultivariateGcdResult(ContractModel):
    gcd: RationalPolynomial
    convention: Literal["MONIC_ASSOCIATE"] = "MONIC_ASSOCIATE"


class MultivariateDivisionRequest(ContractModel):
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


class MultivariateDivisionResult(ContractModel):
    quotient: RationalPolynomial
    remainder: RationalPolynomial
    monomial_order: MonomialOrder
    convention: Literal["EXACT_DIVISION_REMAINDER"] = "EXACT_DIVISION_REMAINDER"


class MultivariateResultantRequest(ContractModel):
    """Compute the resultant of two multivariate polynomials w.r.t. one variable."""

    left: RationalPolynomial
    right: RationalPolynomial
    elimination_variable: PolynomialVariable

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
        return self


class MultivariateScalarValue(ContractModel):
    kind: Literal["SCALAR"] = "SCALAR"
    value: CanonicalRational


class MultivariatePolynomialValue(ContractModel):
    kind: Literal["POLYNOMIAL"] = "POLYNOMIAL"
    value: RationalPolynomial


MultivariateInvariantValue = Annotated[
    MultivariateScalarValue | MultivariatePolynomialValue,
    Field(discriminator="kind"),
]


class MultivariateResultantResult(ContractModel):
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
