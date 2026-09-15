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
from jacobian.math.number_theory.algebraic_numbers.complex import (
    algebraic_root_separation_denominator_bound,
)


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


def _separation_margin(support: sympy.Poly) -> Fraction:
    """A certified rational strictly below half the support's root separation.

    ``algebraic_root_separation_denominator_bound`` bounds the separation of a
    squarefree integer polynomial by ``1/B``; a quarter of that separates
    distinct roots even after a bounded outward expansion.
    """

    _denominator, integral = support.clear_denoms(convert=True)
    _content, primitive = integral.primitive()
    coefficients = tuple(int(coefficient) for coefficient in primitive.all_coeffs())
    bound = algebraic_root_separation_denominator_bound(coefficients)
    return Fraction(1, 4 * bound) if bound > 0 else Fraction(1, 4)


def _complex_support_count(
    support: sympy.Poly, box: tuple[Fraction, Fraction, Fraction, Fraction]
) -> int:
    """Exactly count support roots in a complex box, with a certified retry.

    SymPy's complex staircase count refuses some axis-degenerate boxes such as
    the vertical segment published for a purely imaginary root.  Growing the box
    outward by less than half the separation bound cannot include a distinct
    root, so the retry is exact for the tight boxes produced here.

    A CRootOf/``eval_rational`` enclosure oracle is not a stronger alternative
    here: a closed rectangle for a root such as ``i`` has zero real width with
    the root on its boundary, so every strictly-interior enclosure test is
    undecidable and the oracle would need the same outward expansion.  Keep the
    expansion; if a box is coarser than the separation bound the retry reports
    more than one root, which is a genuine fail-closed result rather than a
    reason to weaken the check.
    """

    def count(
        box_lower: tuple[Fraction, Fraction], box_upper: tuple[Fraction, Fraction]
    ) -> int:
        return int(
            support.count_roots(
                sympy.Rational(box_lower[0]) + sympy.I * sympy.Rational(box_lower[1]),
                sympy.Rational(box_upper[0]) + sympy.I * sympy.Rational(box_upper[1]),
            )
        )

    real_lower, real_upper, imag_lower, imag_upper = box
    try:
        return count((real_lower, imag_lower), (real_upper, imag_upper))
    except NotImplementedError:
        margin = _separation_margin(support)
        try:
            return count(
                (real_lower - margin, imag_lower - margin),
                (real_upper + margin, imag_upper + margin),
            )
        except NotImplementedError as exc:
            raise ValueError(
                "a complex root rectangle has an undecidable support root count"
            ) from exc


def require_rectangle_isolates_one_root(support: sympy.Poly, rectangle: Any) -> None:
    """Assert the rectangle contains exactly one root of ``support``."""

    box = _rectangle_box(rectangle)
    real_lower, real_upper, imag_lower, imag_upper = box
    if imag_lower == 0 and imag_upper == 0:
        if strict_root_count(support, real_lower, real_upper) != 1:
            raise ValueError(
                "a real root rectangle does not isolate exactly one support root"
            )
        return
    if _complex_support_count(support, box) != 1:
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
