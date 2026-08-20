"""Typed wire contracts for polynomial root isolation and algebraic number comparison."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel


class UnivariatePolynomialRequest(StrictModel):
    coefficients_descending: tuple[CanonicalRational, ...] = Field(
        min_length=2, max_length=64
    )

    @model_validator(mode="after")
    def require_nonzero_leading(self) -> Self:
        if self.coefficients_descending[0] == CanonicalRational(num="0", den="1"):
            raise ValueError("leading coefficient must be nonzero")
        return self


class RootIsolationResult(StrictModel):
    """Real roots with rational intervals; rational roots use a singleton."""

    roots: tuple[tuple[CanonicalRational, CanonicalRational], ...]
    multiplicities: tuple[int, ...]
    convention: Literal["SYMPY_REAL_ROOTS"] = "SYMPY_REAL_ROOTS"

    @model_validator(mode="after")
    def require_aligned_intervals(self) -> Self:
        if len(self.roots) != len(self.multiplicities):
            raise ValueError("roots and multiplicities must have the same length")
        if any(
            lower.as_fraction() > upper.as_fraction() for lower, upper in self.roots
        ):
            raise ValueError("isolating intervals must have lower <= upper")
        if any(multiplicity < 1 for multiplicity in self.multiplicities):
            raise ValueError("root multiplicities must be positive")
        return self


class AlgebraicNumberInput(StrictModel):
    polynomial: tuple[CanonicalRational, ...] = Field(min_length=2, max_length=64)
    isolating_interval_lower: CanonicalRational
    isolating_interval_upper: CanonicalRational

    @model_validator(mode="after")
    def require_ordered_interval(self) -> Self:
        if (
            self.isolating_interval_lower.as_fraction()
            >= self.isolating_interval_upper.as_fraction()
        ):
            raise ValueError("isolating interval must have lower < upper")
        if self.polynomial[0].as_fraction() == 0:
            raise ValueError(
                "algebraic-number polynomial must have nonzero leading coefficient"
            )
        from sympy import Poly, Rational, symbols

        x = symbols("x")
        polynomial = Poly(
            sum(
                Rational(*coefficient.as_integer_ratio())
                * x ** (len(self.polynomial) - 1 - index)
                for index, coefficient in enumerate(self.polynomial)
            ),
            x,
        )
        lower = Rational(*self.isolating_interval_lower.as_integer_ratio())
        upper = Rational(*self.isolating_interval_upper.as_integer_ratio())
        roots = {
            root
            for root in polynomial.all_roots()
            if root.is_real and lower <= root <= upper
        }
        if len(roots) != 1:
            raise ValueError("isolating interval must contain exactly one real root")
        return self


class AlgebraicCompareRequest(StrictModel):
    left: AlgebraicNumberInput
    right: AlgebraicNumberInput


class AlgebraicCompareResult(StrictModel):
    order: Literal["LT", "EQ", "GT"]
