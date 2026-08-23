"""Tests for circle inversion of rational planar points."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.math.geometry._models import PointPairRequest, RationalPoint2D
from jacobian.math.geometry.inversion._models import (
    CircleInversionRequest,
    CircleInversionResult,
)
from jacobian.math.geometry.inversion._operations import (
    compute_circle_inversion,
    invert_point,
)
from jacobian.math.geometry.inversion._tools import TOOLS


def _point(x_num, x_den="1", y_num="0", y_den="1"):
    return {
        "x": {"num": x_num, "den": x_den},
        "y": {"num": y_num, "den": y_den},
    }


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
                center=_point("0"),
                power={"num": "0", "den": "1"},
                point=_point("1"),
            )

    def test_center_point_rejected(self) -> None:
        with pytest.raises(ValidationError, match="center cannot be inverted"):
            CircleInversionRequest(
                center=_point("3", "1", "2", "1"),
                power={"num": "5", "den": "1"},
                point=_point("3", "1", "2", "1"),
            )


class TestReusableDomain:
    """Every accepted result must be re-admissible as a request point."""

    def test_result_reenters_the_request_domain(self) -> None:
        first = CircleInversionRequest(
            center=_point("0"),
            power={"num": "1", "den": "1"},
            point=_point("4"),
        )
        result = compute_circle_inversion(first)
        second = CircleInversionRequest(
            center=_point("0"),
            power={"num": "1", "den": "1"},
            point={
                "x": result.inverted_point.x.model_dump(),
                "y": result.inverted_point.y.model_dump(),
            },
        )
        round_trip = compute_circle_inversion(second)
        assert round_trip.inverted_point.x == first.point.x
        assert round_trip.inverted_point.y == first.point.y

    def test_output_beyond_the_reusable_domain_rejected(self) -> None:
        from jacobian.canonical import format_canonical_integer

        n = 10**9000 + 7
        with pytest.raises(ValidationError, match="reusable"):
            CircleInversionRequest(
                center=_point("0"),
                power={"num": "1", "den": "1"},
                point=_point("1", format_canonical_integer(n)),
            )

    def test_admission_is_closed_under_the_returned_point(self) -> None:
        # The reviewer's counterexample: with a 4,000-digit N the first
        # request and its image both fit the half-limit, but re-inverting
        # that image would exceed it under the conservative estimate.  The
        # involutive composition must be decided at admission instead.
        from jacobian.canonical import format_canonical_integer

        n = 10**4000 + 9
        with pytest.raises(ValidationError, match="closed under the returned point"):
            CircleInversionRequest(
                center=_point("0"),
                power={"num": "1", "den": "1"},
                point=_point(format_canonical_integer(n), "1"),
            )


class TestSharedPointValue:
    """The inverted value is the geometry owner's reusable point value."""

    def test_result_feeds_point_consumers_unchanged(self) -> None:
        request = CircleInversionRequest(
            center=_point("0"),
            power={"num": "1", "den": "1"},
            point=_point("4"),
        )
        result = compute_circle_inversion(request)
        assert isinstance(result.inverted_point, RationalPoint2D)
        pair = PointPairRequest(first=result.inverted_point, second=request.point)
        assert pair.first is result.inverted_point

    def test_result_serializes_through_the_shared_type(self) -> None:
        result = compute_circle_inversion(
            CircleInversionRequest(
                center=_point("0"),
                power={"num": "1", "den": "1"},
                point=_point("4"),
            )
        )
        payload = result.model_dump()
        assert payload["inverted_point"] == {
            "x": {"num": "1", "den": "4"},
            "y": {"num": "0", "den": "1"},
        }
        revived = CircleInversionResult.model_validate(payload)
        assert revived == result


class TestWireAdapter:
    def test_compute_circle_inversion(self) -> None:
        request = CircleInversionRequest(
            center=_point("0"),
            power={"num": "1", "den": "1"},
            point=_point("4"),
        )
        result = compute_circle_inversion(request)
        assert result.inverted_point.x.num == "1"
        assert result.inverted_point.x.den == "4"
        assert result.inverted_point.y.num == "0"
        assert result.inverted_point.y.den == "1"

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
