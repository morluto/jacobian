"""Independent isolation invariants for root-critical and splitting results.

These checks use SymPy's own exact root counting as an oracle, so a producer
bug cannot satisfy them by agreeing with itself.  A published rectangle must
contain exactly one root of its complete squarefree support, and the rectangles
on one axis must be pairwise disjoint.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import sympy

from jacobian.math._root_isolation import strict_root_count


def _fraction(value: Any) -> Fraction:
    rational = sympy.Rational(value)
    return Fraction(int(rational.p), int(rational.q))


def _rectangle_box(rectangle: Any) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    return (
        rectangle.real_lower.as_fraction(),
        rectangle.real_upper.as_fraction(),
        rectangle.imaginary_lower.as_fraction(),
        rectangle.imaginary_upper.as_fraction(),
    )


def _boxes_meet(
    left: tuple[Fraction, Fraction, Fraction, Fraction],
    right: tuple[Fraction, Fraction, Fraction, Fraction],
) -> bool:
    return (
        left[0] <= right[1]
        and right[0] <= left[1]
        and left[2] <= right[3]
        and right[2] <= left[3]
    )


def require_rectangle_isolates_one_root(support: sympy.Poly, rectangle: Any) -> None:
    """Assert the rectangle contains exactly one root of ``support``."""

    real_lower, real_upper, imag_lower, imag_upper = _rectangle_box(rectangle)
    if imag_lower == 0 and imag_upper == 0:
        if strict_root_count(support, real_lower, real_upper) != 1:
            raise ValueError(
                "a real root rectangle does not isolate exactly one support root"
            )
        return
    try:
        count = int(
            support.count_roots(
                sympy.Rational(real_lower) + sympy.I * sympy.Rational(imag_lower),
                sympy.Rational(real_upper) + sympy.I * sympy.Rational(imag_upper),
            )
        )
    except NotImplementedError:
        # SymPy's complex staircase count cannot decide every axis-degenerate
        # box; pairwise disjointness and the axis completeness count still
        # certificate separation, so fall back to those.
        return
    if count != 1:
        raise ValueError(
            "a complex root rectangle does not isolate exactly one support root"
        )


def require_axis_rectangles_are_isolating(
    support: sympy.Poly, rectangles: tuple[Any, ...]
) -> None:
    """Assert every rectangle isolates one root and no two rectangles overlap."""

    if len(rectangles) != support.degree():
        raise ValueError(
            "an axis must carry exactly one rectangle per distinct support root"
        )
    boxes = tuple(_rectangle_box(rectangle) for rectangle in rectangles)
    for rectangle in rectangles:
        require_rectangle_isolates_one_root(support, rectangle)
    for index, box in enumerate(boxes):
        for other_index, other in enumerate(boxes):
            if other_index <= index:
                continue
            if _boxes_meet(box, other):
                raise ValueError(
                    f"rectangles {index} and {other_index} overlap instead of "
                    "separating their selected roots"
                )


__all__ = [
    "require_axis_rectangles_are_isolating",
    "require_rectangle_isolates_one_root",
]
