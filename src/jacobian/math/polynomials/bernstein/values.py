"""Source-bound tensor-product Bernstein coordinates over rational boxes."""

from math import prod
from typing import Annotated, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.analysis.intervals import RationalBox
from jacobian.math.polynomials.values import RationalPolynomial

Multidegree = tuple[Annotated[int, Field(ge=0, le=65535)], ...]


class RationalBernsteinPolynomial(StrictModel):
    """Coordinates in product binomial(m_i,k_i)t_i^k_i(1-t_i)^(m_i-k_i).

    Here x_i=a_i+(b_i-a_i)t_i. Coefficients use increasing lexicographic
    multi-indices, with the last axis varying fastest. The complete source
    and box retain the interpretation; serialized coefficients remain claims.
    """

    polynomial: RationalPolynomial
    box: RationalBox
    multidegree: Multidegree = Field(min_length=1, max_length=8)
    coefficients: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=65536)

    @model_validator(mode="after")
    def require_interpretation(self) -> Self:
        if self.polynomial.variables != self.box.variables or len(
            self.multidegree
        ) != len(self.box.variables):
            raise ValueError(
                "Bernstein coordinates require the source's complete ordered axes"
            )
        if prod(m + 1 for m in self.multidegree) != len(self.coefficients):
            raise ValueError("Bernstein tensor shape must match the multidegree")
        if any(interval.lower == interval.upper for interval in self.box.intervals):
            raise ValueError("Bernstein boxes require strictly positive widths")
        if any(
            any(e > m for e, m in zip(term.exponents, self.multidegree, strict=True))
            for term in self.polynomial.polynomial.terms
        ):
            raise ValueError(
                "Bernstein multidegree must dominate source coordinate degrees"
            )
        return self


__all__ = ["RationalBernsteinPolynomial"]
