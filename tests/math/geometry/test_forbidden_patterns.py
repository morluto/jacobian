"""Tests for the forbidden-patterns operation and its source-bound replay."""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.geometry._models import (
    ForbiddenConfiguration,
    ForbiddenLabelledPoint,
    ForbiddenPatternsRequest,
    RationalPoint2D,
)
from jacobian.math.geometry._operations import forbidden_patterns


def _point(label: str, x: Fraction, y: Fraction) -> ForbiddenLabelledPoint:
    return ForbiddenLabelledPoint(
        label=label,
        point=RationalPoint2D(
            x=CanonicalRational.from_integer_ratio(x.numerator, x.denominator),
            y=CanonicalRational.from_integer_ratio(y.numerator, y.denominator),
        ),
    )


def _run(points):
    request = ForbiddenPatternsRequest(
        configuration=ForbiddenConfiguration(points=tuple(points))
    )
    return forbidden_patterns(request)


class TestForbiddenPatterns:
    def test_collinear_triple_with_distinct_row_scales(self):
        """(0,1), (1/2,3/2), (1/3,4/3) clear to rows with distinct scales;
        the replay must test collinearity on the full (X, Y, D) rows."""
        result = _run(
            (
                _point("A", Fraction(0), Fraction(1)),
                _point("B", Fraction(1, 2), Fraction(3, 2)),
                _point("C", Fraction(1, 3), Fraction(4, 3)),
            )
        )
        assert result.has_collinear_triple
        assert (
            result.collinear_triple.first,
            result.collinear_triple.second,
            result.collinear_triple.third,
        ) == (0, 1, 2)
        assert not result.has_concyclic_quadruple

    def test_triangle_has_no_forbidden_pattern(self):
        result = _run(
            (
                _point("A", Fraction(0), Fraction(0)),
                _point("B", Fraction(1), Fraction(0)),
                _point("C", Fraction(0), Fraction(1)),
            )
        )
        assert not result.has_collinear_triple
        assert not result.has_concyclic_quadruple
        assert result.checked_triples == 1
        assert result.checked_quadruples == 0

    def test_concyclic_unit_square(self):
        result = _run(
            (
                _point("A", Fraction(0), Fraction(0)),
                _point("B", Fraction(1), Fraction(0)),
                _point("C", Fraction(1), Fraction(1)),
                _point("D", Fraction(0), Fraction(1)),
            )
        )
        assert not result.has_collinear_triple
        assert result.has_concyclic_quadruple
        assert (
            result.concyclic_quadruple.first,
            result.concyclic_quadruple.second,
            result.concyclic_quadruple.third,
            result.concyclic_quadruple.fourth,
        ) == (0, 1, 2, 3)

    def test_collinear_quadruple_is_not_a_concyclic_witness(self):
        """Four collinear points satisfy no nondegenerate-circle determinant
        witness; degeneracy must be replayed on full homogeneous rows too."""
        result = _run(
            (
                _point("A", Fraction(0), Fraction(0)),
                _point("B", Fraction(1, 2), Fraction(1, 2)),
                _point("C", Fraction(1), Fraction(1)),
                _point("D", Fraction(2), Fraction(2)),
            )
        )
        assert result.has_collinear_triple
        assert not result.has_concyclic_quadruple
