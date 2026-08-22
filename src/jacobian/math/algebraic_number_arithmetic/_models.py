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

    @model_validator(mode="after")
    def require_result_within_operand_bound(self) -> Self:
        """Narrow admission so every accepted request's exact result fits the 256-digit contract.

        Both addition and multiplication are closed under this check: the
        validator computes the exact ``add`` and ``multiply`` results as
        Fractions and verifies that each rational component survives the
        canonical 256-digit validator.  This guarantees the operation can
        return its declared ``AlgebraicArithmeticResult`` (which reuses the
        same 256-digit bound) without exposing a host ``ValidationError``.
        """
        from fractions import Fraction

        from jacobian._exact import CanonicalRational

        def _fits(frac: Fraction) -> bool:
            # Check digit bound without constructing the full model twice
            num_digits = len(str(abs(frac.numerator)))
            den_digits = len(str(frac.denominator))
            if num_digits > 256 or den_digits > 256:
                return False
            # Also ensure canonical reduction would succeed (it will if digits fit)
            try:
                CanonicalRational.from_fraction(frac)
            except ValueError:
                return False
            return True

        a, b = self.left.rational_part.as_fraction(), self.left.radical_coefficient.as_fraction()
        c, e = self.right.rational_part.as_fraction(), self.right.radical_coefficient.as_fraction()
        d = self.left.radicand
        # Addition: (a+c) + (b+e)*sqrt(d)
        if not _fits(a + c) or not _fits(b + e):
            raise ValueError(
                "operands would produce an addition result exceeding the 256-digit canonical rational bound"
            )
        # Multiplication: (ac + bed) + (ae + bc)*sqrt(d)
        if not _fits(a * c + b * e * d) or not _fits(a * e + b * c):
            raise ValueError(
                "operands would produce a multiplication result exceeding the 256-digit canonical rational bound"
            )
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
