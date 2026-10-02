"""Contracts for exact multivariate polynomial division over ``QQ``."""

from __future__ import annotations

from typing import Any, Literal, Self

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

MonomialOrder = Literal["lex", "grlex", "grevlex"]
"""Declared monomial order for multivariate polynomial division."""


class MultivariateDivisionRequest(StrictModel):
    """Divide one polynomial by another under a declared monomial order.

    Sources retain their full ordered QQ ring. Before expansion, division
    admits the coefficient-free leading-monomial rewrite closure, bounded
    quotient and remainder supports (each at most 4,096 terms), exact scalar
    growth, intermediate storage, reconstruction work, and complete output.
    Canceling paths can make these conservative envelopes larger than the
    actual result. One-variable inputs use the classical degree envelope.
    """

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


def _polynomial_from_ring(poly: Any, variables: tuple[str, ...]) -> RationalPolynomial:
    """Convert an admitted QQ ring element without a dense or expression pass."""
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_integer_ratio(
                        int(coefficient.numerator), int(coefficient.denominator)
                    ),
                    exponents=exponents,
                )
                # Wire terms are always lexicographic, independently of the
                # monomial order used by the division kernel.
                for exponents, coefficient in sorted(poly.items(), reverse=True)
            )
        ),
    )


__all__ = [
    "MonomialOrder",
    "MultivariateDivisionRequest",
    "MultivariateDivisionResult",
]
