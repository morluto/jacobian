"""Contracts for multivariate polynomial GCD over ``QQ``."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomials.multivariate._models import (
    _MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
    _MAX_MULTIVARIATE_EXPONENT,
    _MAX_MULTIVARIATE_TERMS,
    _validate_multivariate_pair,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    require_polynomial_budget,
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


__all__ = ["MultivariateGcdRequest", "MultivariateGcdResult"]
