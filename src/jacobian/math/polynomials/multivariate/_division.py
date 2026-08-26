"""Contracts for exact multivariate polynomial division over ``QQ``."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomials.multivariate._models import (
    _MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
    _MAX_MULTIVARIATE_EXPONENT,
    _MAX_MULTIVARIATE_TERMS,
    _validate_multivariate_pair,
    _validation_error,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    require_polynomial_budget,
)

MonomialOrder = Literal["lex", "grlex", "grevlex"]
"""Declared monomial order for multivariate polynomial division."""


class MultivariateDivisionRequest(StrictModel):
    """Divide one multivariate polynomial by another under a declared monomial order."""

    left: RationalPolynomial
    right: RationalPolynomial
    monomial_order: MonomialOrder = "lex"

    @model_validator(mode="after")
    def require_multivariate_ring(self) -> Self:
        _validate_multivariate_pair(self.left, self.right)
        if not self.right.polynomial.terms:
            raise _validation_error("divisor polynomial must be nonzero")
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


__all__ = [
    "MonomialOrder",
    "MultivariateDivisionRequest",
    "MultivariateDivisionResult",
]
