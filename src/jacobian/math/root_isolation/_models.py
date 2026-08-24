"""Typed wire contracts for polynomial root isolation and algebraic number comparison."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.real_algebraic import (
    RealAlgebraicOrderValue,
    RealAlgebraicValue,
)


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


AlgebraicNumberInput = RealAlgebraicValue


class AlgebraicCompareRequest(StrictModel):
    left: AlgebraicNumberInput
    right: AlgebraicNumberInput


AlgebraicCompareResult = RealAlgebraicOrderValue


__all__ = [
    "AlgebraicCompareRequest",
    "AlgebraicCompareResult",
    "AlgebraicNumberInput",
    "RootIsolationResult",
    "UnivariatePolynomialRequest",
]
