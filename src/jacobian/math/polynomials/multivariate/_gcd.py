"""Contracts for multivariate polynomial GCD over ``QQ``."""

from __future__ import annotations

from typing import Literal, Self

from jacobian._models import StrictModel
from jacobian.math.polynomials.values import RationalPolynomial


class MultivariateGcdRequest(StrictModel):
    """Two multivariate polynomials in ``QQ[x_1, ..., x_n]``."""

    left: RationalPolynomial
    right: RationalPolynomial


class MultivariateGcdResult(StrictModel):
    left: RationalPolynomial
    right: RationalPolynomial
    gcd: RationalPolynomial
    convention: Literal["MONIC_ASSOCIATE"] = "MONIC_ASSOCIATE"

    @classmethod
    def _from_kernel(
        cls,
        left: RationalPolynomial,
        right: RationalPolynomial,
        gcd: RationalPolynomial,
    ) -> Self:
        return cls.model_construct(left=left, right=right, gcd=gcd)


__all__ = ["MultivariateGcdRequest", "MultivariateGcdResult"]
