"""Contracts for exact multivariate polynomial division over ``QQ``."""

from __future__ import annotations

from typing import Literal, Self

from jacobian._models import StrictModel
from jacobian.math.polynomials.values import RationalPolynomial

MonomialOrder = Literal["lex", "grlex", "grevlex"]
"""Declared monomial order for multivariate polynomial division."""


class MultivariateDivisionRequest(StrictModel):
    """Divide one multivariate polynomial by another under a declared monomial order."""

    left: RationalPolynomial
    right: RationalPolynomial
    monomial_order: MonomialOrder = "lex"


class MultivariateDivisionResult(StrictModel):
    left: RationalPolynomial
    right: RationalPolynomial
    quotient: RationalPolynomial
    remainder: RationalPolynomial
    monomial_order: MonomialOrder
    convention: Literal["EXACT_DIVISION_REMAINDER"] = "EXACT_DIVISION_REMAINDER"

    @classmethod
    def _from_kernel(
        cls,
        *,
        left: RationalPolynomial,
        right: RationalPolynomial,
        quotient: RationalPolynomial,
        remainder: RationalPolynomial,
        monomial_order: MonomialOrder,
    ) -> Self:
        return cls.model_construct(
            left=left,
            right=right,
            quotient=quotient,
            remainder=remainder,
            monomial_order=monomial_order,
        )


__all__ = [
    "MonomialOrder",
    "MultivariateDivisionRequest",
    "MultivariateDivisionResult",
]
