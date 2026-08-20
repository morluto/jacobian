"""Conservative decimal-height propagation for bounded exact rational arithmetic."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from jacobian._exact import CanonicalRational


@dataclass(frozen=True)
class RationalHeight:
    """Upper bounds for reduced numerator and denominator decimal digits."""

    numerator_digits: int
    denominator_digits: int

    @classmethod
    def from_canonical(cls, value: CanonicalRational) -> RationalHeight:
        return cls(len(value.num.lstrip("-")), len(value.den))

    def product(self, other: RationalHeight) -> RationalHeight:
        return RationalHeight(
            self.numerator_digits + other.numerator_digits,
            self.denominator_digits + other.denominator_digits,
        )

    def quotient(self, other: RationalHeight) -> RationalHeight:
        return RationalHeight(
            self.numerator_digits + other.denominator_digits,
            self.denominator_digits + other.numerator_digits,
        )

    def exceeds(self, max_digits: int) -> bool:
        return (
            self.numerator_digits > max_digits or self.denominator_digits > max_digits
        )


def sum_heights(values: Iterable[RationalHeight]) -> RationalHeight:
    """Bound a rational sum using a product common denominator.

    Reduction can only decrease either component.  For ``m`` summands, the
    product denominator has at most the sum of denominator digit bounds.  Each
    lifted numerator has its own numerator bound plus every other denominator
    bound, and adding ``m`` such integers costs at most ``digits(m)`` more.
    """
    items = tuple(values)
    if not items:
        return RationalHeight(1, 1)
    denominator_digits = sum(item.denominator_digits for item in items)
    numerator_digits = max(
        item.numerator_digits + denominator_digits - item.denominator_digits
        for item in items
    ) + len(str(len(items)))
    return RationalHeight(numerator_digits, denominator_digits)


__all__ = ["RationalHeight", "sum_heights"]
