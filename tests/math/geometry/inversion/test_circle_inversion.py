"""Tests for circle inversion of rational planar points."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.math.geometry._models import RationalPoint2D
from jacobian.math.geometry.inversion._models import (
    CircleInversionRequest,
    CircleInversionResult,
)
from jacobian.math.geometry.inversion._operations import (
    compute_circle_inversion,
    invert_point,
)
from jacobian.math.geometry.inversion._tools import TOOLS


def _point(x_num: str, x_den: str, y_num: str, y_den: str) -> RationalPoint2D:
    return RationalPoint2D(
        x={"num": x_num, "den": x_den},
        y={"num": y_num, "den": y_den},
    )


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
                center=_point("0", "1", "0", "1"),
                power={"num": "0", "den": "1"},
                point=_point("1", "1", "0", "1"),
            )

    def test_oversized_input_rejected(self) -> None:
        big = "9" * 17000
        with pytest.raises(ValidationError, match="reusable admission bound"):
            CircleInversionRequest(
                center=_point("0", "1", "0", "1"),
                power={"num": "1", "den": "1"},
                point=_point(big, "1", "0", "1"),
            )

    def test_output_beyond_the_reusable_bound_rejected(self) -> None:
        """An input inside the bound whose exact image leaves it is rejected."""
        from jacobian.canonical import format_canonical_integer

        big = 10**9000 + 1
        request = {
            "center": _point("0", "1", "0", "1"),
            "power": {"num": "1", "den": "1"},
            # (big, 1): image is ~(1/big, 1/big^2); the second coordinate's
            # denominator has ~18000 digits, beyond the half-limit bound.
            "point": _point(format_canonical_integer(big), "1", "1", "1"),
        }
        with pytest.raises(ValidationError, match="reusable admission bound"):
            CircleInversionRequest.model_validate(request)


class TestInvolutionClosure:
    """Admission accounts for the defining involution exactly."""

    def test_reduction_heavy_pair_is_admitted_and_closed(self) -> None:
        """The triage counterexample: p = (1/q1, 1/q2) with ~1401-digit q.

        The cancellation-blind estimator measured the re-inverted height as
        ~16,816 digits and rejected the feed-back even though the exact
        inverse is the original admitted point. Exact admission must admit
        both directions.
        """
        q1 = 10**1400 + 1
        q2 = 10**1400 + 3
        request = CircleInversionRequest(
            center=_point("0", "1", "0", "1"),
            power={"num": "1", "den": "1"},
            point=_point("1", str(q1), "1", str(q2)),
        )
        result = compute_circle_inversion(request)

        feedback = CircleInversionRequest(
            center=result.center,
            power=result.power,
            point=result.image,
        )
        round_trip = compute_circle_inversion(feedback)
        assert round_trip.image.x.as_fraction() == Fraction(1, q1)
        assert round_trip.image.y.as_fraction() == Fraction(1, q2)

    def test_image_is_the_shared_geometry_value(self) -> None:
        result = compute_circle_inversion(
            CircleInversionRequest(
                center=_point("0", "1", "0", "1"),
                power={"num": "1", "den": "1"},
                point=_point("4", "1", "0", "1"),
            )
        )
        assert isinstance(result.image, RationalPoint2D)
        assert result.image.x.num == "1"
        assert result.image.x.den == "4"
        payload = result.model_dump(mode="json")
        reparsed = CircleInversionResult.model_validate(payload)
        assert reparsed.image == result.image


class TestWireAdapter:
    def test_compute_circle_inversion(self) -> None:
        request = CircleInversionRequest(
            center=_point("0", "1", "0", "1"),
            power={"num": "1", "den": "1"},
            point=_point("4", "1", "0", "1"),
        )
        result = compute_circle_inversion(request)
        assert result.image.x.num == "1"
        assert result.image.x.den == "4"
        assert result.image.y.num == "0"
        assert result.image.y.den == "1"

    def test_non_origin_center(self) -> None:
        request = CircleInversionRequest(
            center=_point("3", "1", "2", "1"),
            power={"num": "5", "den": "1"},
            point=_point("1", "1", "7", "1"),
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
