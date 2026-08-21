"""Typed wire contracts for algebraic number arithmetic."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel

_MAX_COEFFICIENT_DIGITS = 256
_MAX_SQUAREFREE_RADICAND = 10_000


class QuadraticElement(StrictModel):
    """An element a + b*sqrt(d) of Q(sqrt(d))."""

    rational_part_num: int = Field(description="Numerator of the rational part a.")
    rational_part_den: int = Field(default=1, ge=1)
    irrational_part_num: int = Field(
        description="Numerator of the coefficient of sqrt(d)."
    )
    irrational_part_den: int = Field(default=1, ge=1)
    radicand: StrictInt = Field(
        description="The squarefree integer d under the radical."
    )

    @model_validator(mode="after")
    def require_valid_radicand(self) -> Self:
        if self.radicand < 2:
            raise ValueError("radicand must be at least 2")
        if self.radicand > _MAX_SQUAREFREE_RADICAND:
            raise ValueError("radicand exceeds the supported bound")
        for label, num, den in (
            ("rational_part", self.rational_part_num, self.rational_part_den),
            ("irrational_part", self.irrational_part_num, self.irrational_part_den),
        ):
            if abs(num) >= 10**_MAX_COEFFICIENT_DIGITS:
                raise ValueError(f"{label} numerator exceeds the digit bound")
            if den >= 10**_MAX_COEFFICIENT_DIGITS:
                raise ValueError(f"{label} denominator exceeds the digit bound")
        return self


class AlgebraicArithmeticRequest(StrictModel):
    """Two elements of Q(sqrt(d)) for an exact binary operation."""

    left: QuadraticElement
    right: QuadraticElement

    @model_validator(mode="after")
    def require_same_radicand(self) -> Self:
        if self.left.radicand != self.right.radicand:
            raise ValueError("both operands must belong to the same quadratic field")
        return self


class AlgebraicArithmeticResult(StrictModel):
    """The exact result element a + b*sqrt(d) of one operation."""

    rational_part_num: int
    rational_part_den: int
    irrational_part_num: int
    irrational_part_den: int
    radicand: StrictInt

    @model_validator(mode="after")
    def require_normalized(self) -> Self:
        if self.rational_part_den <= 0:
            raise ValueError("rational part denominator must be positive")
        if self.irrational_part_den <= 0:
            raise ValueError("irrational part denominator must be positive")
        return self


__all__ = [
    "AlgebraicArithmeticRequest",
    "AlgebraicArithmeticResult",
    "QuadraticElement",
]
