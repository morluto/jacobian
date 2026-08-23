"""Tests for the circumradius profile operation and its height admission."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry._models import (
    CircumradiusProfileRequest,
    LabelledPoint2D,
    RationalPoint2D,
)
from jacobian.math.geometry._operations import circumradius_profile


def _rational(num: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(num, den))


def _point(label: str, x: CanonicalRational, y: CanonicalRational) -> LabelledPoint2D:
    return LabelledPoint2D(label=label, point=RationalPoint2D(x=x, y=y))


class TestCircumradiusProfile:
    def test_unit_right_triangle(self):
        request = CircumradiusProfileRequest(
            points=(
                _point("A", _rational(0), _rational(0)),
                _point("B", _rational(1), _rational(0)),
                _point("C", _rational(0), _rational(1)),
            )
        )
        result = circumradius_profile(request)
        assert result.entries[0].squared_circumradius.as_fraction() == Fraction(1, 2)
        assert not result.entries[0].collinear

    def test_rational_right_triangle_matches_hypotenuse(self):
        """The right angle sits at A, so R^2 = |BC|^2 / 4."""
        request = CircumradiusProfileRequest(
            points=(
                _point("A", _rational(0), _rational(0)),
                _point("B", _rational(1, 2), _rational(0)),
                _point("C", _rational(0), _rational(1, 3)),
            )
        )
        result = circumradius_profile(request)
        assert result.entries[0].squared_circumradius.as_fraction() == Fraction(13, 144)

    def test_collinear_triple_has_no_radius(self):
        request = CircumradiusProfileRequest(
            points=(
                _point("A", _rational(0), _rational(0)),
                _point("B", _rational(1), _rational(0)),
                _point("C", _rational(2), _rational(0)),
            )
        )
        result = circumradius_profile(request)
        assert result.entries[0].collinear
        assert result.entries[0].squared_circumradius is None


class TestCircumradiusHeightAdmission:
    def test_distinct_4096_digit_denominators_rejected(self):
        """Independent near-limit denominators multiply past the result bound;
        the previous flat per-coordinate cap accepted such configurations."""
        from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS

        base = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 28_673)
        assert len(str(base)) == 4096
        points = tuple(
            _point(
                chr(65 + index // 2),
                _rational(1, base + index),
                _rational(1, base - index),
            )
            for index in range(0, 6, 2)
        )
        with pytest.raises(ValidationError, match="canonical result bound"):
            CircumradiusProfileRequest(points=points)

    def test_exact_4096_digit_denominators_rejected(self):
        from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS

        denominator = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 28_673) + 1
        assert len(str(denominator)) == 4096
        points = tuple(
            _point(
                chr(65 + i),
                _rational(1, denominator + i),
                _rational(1, denominator - i),
            )
            for i in range(3)
        )
        with pytest.raises(ValidationError, match="canonical result bound"):
            CircumradiusProfileRequest(points=points)

    def test_moderate_heights_accepted_and_executed(self):
        denominator = 10**700 + 3
        points = tuple(
            _point(
                chr(65 + i),
                _rational(1, denominator + i),
                _rational(1, denominator - i),
            )
            for i in range(3)
        )
        request = CircumradiusProfileRequest(points=points)
        result = circumradius_profile(request)
        value = result.entries[0].squared_circumradius
        assert value is not None
        digits = max(len(value.num), len(value.den))
        from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS

        assert digits <= MAX_CANONICAL_RATIONAL_DIGITS
