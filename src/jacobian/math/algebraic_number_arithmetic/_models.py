"""Typed wire contracts for algebraic number arithmetic."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.real_quadratic import RealQuadraticValue

# Reuse the canonical quadratic field value so arithmetic results compose
# directly with ``arithmetic.real_quadratic.order.compute`` without field
# reconstruction.  ``QuadraticElement`` is retained as an alias for
# backwards compatibility (P1).
QuadraticElement = RealQuadraticValue


class AlgebraicArithmeticRequest(StrictModel):
    """Two elements of Q(sqrt(d)) for an exact binary operation."""

    left: RealQuadraticValue
    right: RealQuadraticValue

    @model_validator(mode="after")
    def require_same_radicand(self) -> Self:
        if self.left.radicand != self.right.radicand:
            raise ValueError("both operands must belong to the same quadratic field")
        return self


class AlgebraicArithmeticResult(RealQuadraticValue):
    """The exact result element a + b*sqrt(d) of one operation.

    Subclassing ``RealQuadraticValue`` reuses its square-free radicand
    enforcement (P2 line 33) and 256-digit bounded-rational validation,
    guaranteeing every returned element remains consumable as a subsequent
    operand (P2 line 41).
    """


__all__ = [
    "AlgebraicArithmeticRequest",
    "AlgebraicArithmeticResult",
    "QuadraticElement",
]
