"""Tests for circle inversion of rational planar points."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.math.geometry.inversion._models import (
    CircleInversionRequest,
    CircleInversionResult,
)
from jacobian.math.geometry.inversion._operations import (
    compute_circle_inversion,
    invert_point,
)
from jacobian.math.geometry.inversion._tools import TOOLS


class TestKnownAnswers:
    """Exact fixtures from issue #2121."""

    def test_unit_inversion_at_origin(self) -> None:
        # B = (4,0) -> (1/4, 0)
        qx, qy = invert_point(Fraction(0), Fraction(0), Fraction(1), Fraction(4), Fraction(0))
        assert qx == Fraction(1, 4)
        assert qy == Fraction(0)

    def test_center_inversion(self) -> None:
        # C = (1,2) -> (1/5, 2/5)
        qx, qy = invert_point(Fraction(0), Fraction(0), Fraction(1), Fraction(1), Fraction(2))
        assert qx == Fraction(1, 5)
        assert qy == Fraction(2, 5)

    def test_halfplane_inversion(self) -> None:
        # H = (1, 3/2) -> (4/13, 6/13)
        qx, qy = invert_point(Fraction(0), Fraction(0), Fraction(1), Fraction(1), Fraction(3, 2))
        assert qx == Fraction(4, 13)
        assert qy == Fraction(6, 13)


class TestInvariance:
    def test_involutivity(self) -> None:
        """I_{c,s}(I_{c,s}(p)) = p."""
        cx, cy = Fraction(3), Fraction(2)
        s = Fraction(5)
        px, py = Fraction(1), Fraction(7)
        qx, qy = invert_point(cx, cy, s, px, py)
        rx, ry = invert_point(cx, cy, s, qx, qy)
        assert rx == px
        assert ry == py

    def test_involutivity_non_unit_power(self) -> None:
        cx, cy = Fraction(0), Fraction(0)
        s = Fraction(9)
        px, py = Fraction(3), Fraction(0)
        qx, qy = invert_point(cx, cy, s, px, py)
        rx, ry = invert_point(cx, cy, s, qx, qy)
        assert rx == px
        assert ry == py

    def test_preserves_inversion_power_product(self) -> None:
        """||p-c||² * ||q-c||² = s²."""
        cx, cy = Fraction(1), Fraction(1)
        s = Fraction(4)
        px, py = Fraction(3), Fraction(2)
        qx, qy = invert_point(cx, cy, s, px, py)
        norm_p = (px - cx) ** 2 + (py - cy) ** 2
        norm_q = (qx - cx) ** 2 + (qy - cy) ** 2
        assert norm_p * norm_q == s * s


class TestRejection:
    def test_center_rejected(self) -> None:
        with pytest.raises(ValueError, match="center"):
            invert_point(Fraction(0), Fraction(0), Fraction(1), Fraction(0), Fraction(0))

    def test_zero_power_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CircleInversionRequest(
                center_x={"num": "0", "den": "1"},
                center_y={"num": "0", "den": "1"},
                power={"num": "0", "den": "1"},
                point_x={"num": "1", "den": "1"},
                point_y={"num": "0", "den": "1"},
            )


class TestWireAdapter:
    def test_compute_circle_inversion(self) -> None:
        request = CircleInversionRequest(
            center_x={"num": "0", "den": "1"},
            center_y={"num": "0", "den": "1"},
            power={"num": "1", "den": "1"},
            point_x={"num": "4", "den": "1"},
            point_y={"num": "0", "den": "1"},
        )
        result = compute_circle_inversion(request)
        assert result.inverted_point.x.num == "1"
        assert result.inverted_point.x.den == "4"
        assert result.inverted_point.y.num == "0"
        assert result.inverted_point.y.den == "1"

    def test_non_origin_center(self) -> None:
        request = CircleInversionRequest(
            center_x={"num": "3", "den": "1"},
            center_y={"num": "2", "den": "1"},
            power={"num": "5", "den": "1"},
            point_x={"num": "1", "den": "1"},
            point_y={"num": "7", "den": "1"},
        )
        result = compute_circle_inversion(request)
        assert isinstance(result, CircleInversionResult)


class TestDigitBounds:
    """The admitted domain must guarantee exact results within canonical limits."""

    @staticmethod
    def _reciprocal(d: int) -> CanonicalRational:
        return CanonicalRational.from_integer_ratio(1, d)

    def test_result_overflow_inputs_rejected_at_admission(self) -> None:
        big_den = 10**7999
        with pytest.raises(ValidationError, match="2730-digit bound"):
            CircleInversionRequest(
                center_x=self._reciprocal(big_den + 3),
                center_y=self._reciprocal(big_den + 7),
                power=self._reciprocal(big_den + 11),
                point_x=self._reciprocal(big_den + 13),
                point_y=self._reciprocal(big_den + 17),
            )

    def test_boundary_inputs_return_exact_result(self) -> None:
        big_den = 10**2729
        request = CircleInversionRequest(
            center_x=CanonicalRational.from_integer_ratio(
                10**2728 + 11, big_den
            ),
            center_y=CanonicalRational.from_integer_ratio(
                10**2728 + 17, big_den - 5
            ),
            power=CanonicalRational.from_integer_ratio(
                big_den - 7, 10**2728 + 23
            ),
            point_x=CanonicalRational.from_integer_ratio(
                10**2728 + 29, big_den - 11
            ),
            point_y=CanonicalRational.from_integer_ratio(
                big_den - 13, 10**2728 + 31
            ),
        )
        result = compute_circle_inversion(request)
        qx, qy = result.inverted_point.x, result.inverted_point.y
        assert len(qx.num) <= MAX_CANONICAL_RATIONAL_DIGITS
        assert len(qx.den) <= MAX_CANONICAL_RATIONAL_DIGITS
        assert len(qy.num) <= MAX_CANONICAL_RATIONAL_DIGITS
        assert len(qy.den) <= MAX_CANONICAL_RATIONAL_DIGITS
        cx = request.center_x.as_fraction()
        cy = request.center_y.as_fraction()
        s = request.power.as_fraction()
        px = request.point_x.as_fraction()
        py = request.point_y.as_fraction()
        norm_p = (px - cx) ** 2 + (py - cy) ** 2
        norm_q = (
            qx.as_fraction() - cx
        ) ** 2 + (
            qy.as_fraction() - cy
        ) ** 2
        assert norm_p * norm_q == s * s


class TestToolsAndExamples:
    def test_one_tool(self) -> None:
        assert len(TOOLS) == 1

    @pytest.mark.parametrize(
        "tool",
        TOOLS,
        ids=[t.operation_id for t in TOOLS],
    )
    def test_examples_run(self, tool) -> None:
        for ex in tool.examples:
            request = tool.request_type.model_validate(ex.input)
            result = tool.run(request)
            assert result is not None
