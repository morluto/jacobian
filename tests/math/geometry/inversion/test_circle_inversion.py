"""Tests for circle inversion of rational planar points."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

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
        qx, qy = invert_point(
            Fraction(0), Fraction(0), Fraction(1), Fraction(4), Fraction(0)
        )
        assert qx == Fraction(1, 4)
        assert qy == Fraction(0)

    def test_center_inversion(self) -> None:
        # C = (1,2) -> (1/5, 2/5)
        qx, qy = invert_point(
            Fraction(0), Fraction(0), Fraction(1), Fraction(1), Fraction(2)
        )
        assert qx == Fraction(1, 5)
        assert qy == Fraction(2, 5)

    def test_halfplane_inversion(self) -> None:
        # H = (1, 3/2) -> (4/13, 6/13)
        qx, qy = invert_point(
            Fraction(0), Fraction(0), Fraction(1), Fraction(1), Fraction(3, 2)
        )
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
            invert_point(
                Fraction(0), Fraction(0), Fraction(1), Fraction(0), Fraction(0)
            )

    def test_zero_power_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CircleInversionRequest(
                center={"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
                power={"num": "0", "den": "1"},
                point={"x": {"num": "1", "den": "1"}, "y": {"num": "0", "den": "1"}},
            )

    def test_admission_closed_under_returned_point(self) -> None:
        """A point whose image would not be reusable as a later input is
        rejected: estimated output heights must stay within the reusable
        quarter-limit cap, and the exact image must satisfy the predicate."""
        a = 2**7500
        with pytest.raises(ValidationError, match="height bound"):
            CircleInversionRequest(
                center={"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
                power={"num": "1", "den": "1"},
                point={
                    "x": {"num": "1", "den": str(a)},
                    "y": {"num": "1", "den": str(3**7500)},
                },
            )

    def test_image_of_admitted_point_is_admitted(self) -> None:
        """Every accepted request's exact image satisfies the same request."""
        request = CircleInversionRequest(
            center={"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
            power={"num": "1", "den": "1"},
            point={"x": {"num": "3", "den": "5"}, "y": {"num": "-1", "den": "2"}},
        )
        image = compute_circle_inversion(request).inverted
        refed = CircleInversionRequest(
            center=request.center,
            power=request.power,
            point={
                "x": {"num": image.x.num, "den": image.x.den},
                "y": {"num": image.y.num, "den": image.y.den},
            },
        )
        # Admission accepts the exact image, and inverting it returns the
        # original point (involution).
        round_trip = compute_circle_inversion(refed).inverted
        assert round_trip.x.as_fraction() == request.point.x.as_fraction()
        assert round_trip.y.as_fraction() == request.point.y.as_fraction()


class TestWireAdapter:
    def test_compute_circle_inversion(self) -> None:
        request = CircleInversionRequest(
            center={"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
            power={"num": "1", "den": "1"},
            point={"x": {"num": "4", "den": "1"}, "y": {"num": "0", "den": "1"}},
        )
        result = compute_circle_inversion(request)
        assert result.inverted.x.num == "1"
        assert result.inverted.x.den == "4"
        assert result.inverted.y.num == "0"
        assert result.inverted.y.den == "1"

    def test_non_origin_center(self) -> None:
        request = CircleInversionRequest(
            center={"x": {"num": "3", "den": "1"}, "y": {"num": "2", "den": "1"}},
            power={"num": "5", "den": "1"},
            point={"x": {"num": "1", "den": "1"}, "y": {"num": "7", "den": "1"}},
        )
        result = compute_circle_inversion(request)
        assert isinstance(result, CircleInversionResult)


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
