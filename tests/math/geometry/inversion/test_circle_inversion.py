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
        assert result.point.x.num == "1"
        assert result.point.x.den == "4"
        assert result.point.y.num == "0"
        assert result.point.y.den == "1"
        CircleInversionResult.model_validate(result.model_dump(mode="json"))

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


class TestCanonicalPointComposition:
    def test_serialized_point_feeds_pair_operations_unchanged(self) -> None:
        """The retained point is the canonical geometry point value."""
        from jacobian.math.geometry._models import PointPairRequest

        request = CircleInversionRequest(
            center_x={"num": "0", "den": "1"},
            center_y={"num": "0", "den": "1"},
            power={"num": "4", "den": "1"},
            point_x={"num": "2", "den": "1"},
            point_y={"num": "0", "den": "1"},
        )
        result = compute_circle_inversion(request)
        payload = result.model_dump(mode="json")
        pair = PointPairRequest.model_validate(
            {"first": payload["point"], "second": {"x": {"num": "1", "den": "1"}, "y": {"num": "1", "den": "1"}}}
        )
        assert pair.first.x.num == "2"
        assert pair.first.y.num == "0"

    def test_tampered_point_rejected_by_replay(self) -> None:
        request = CircleInversionRequest(
            center_x={"num": "0", "den": "1"},
            center_y={"num": "0", "den": "1"},
            power={"num": "1", "den": "1"},
            point_x={"num": "4", "den": "1"},
            point_y={"num": "0", "den": "1"},
        )
        result = compute_circle_inversion(request)
        payload = result.model_dump()
        payload["point"]["x"]["num"] = "9"
        with pytest.raises(ValidationError):
            CircleInversionResult.model_validate(payload)


def test_derived_digit_count_is_limit_independent() -> None:
    """Inputs inside every declared bound validate without int->str limits.

    With N = 10^4095 every input component has at most 4,096 digits and the
    inverse coordinate N^2 carries 8,191 digits, which crosses CPython's
    default str(int) conversion limit; counting digits via the canonical
    formatter must admit this request instead of raising before the explicit
    canonical-range check can run.
    """
    big = "1" + "0" * 4095
    request = CircleInversionRequest.model_validate(
        {
            "center_x": {"num": "0", "den": "1"},
            "center_y": {"num": "0", "den": "1"},
            "power": {"num": big, "den": "1"},
            "point_x": {"num": "1", "den": big},
            "point_y": {"num": "0", "den": "1"},
        }
    )
    assert request.power.num == big


def test_extreme_power_inverts_to_canonical_point() -> None:
    """The extreme admitted request composes through to its typed result."""
    big = "1" + "0" * 4095
    request = CircleInversionRequest.model_validate(
        {
            "center_x": {"num": "0", "den": "1"},
            "center_y": {"num": "0", "den": "1"},
            "power": {"num": big, "den": "1"},
            "point_x": {"num": "1", "den": big},
            "point_y": {"num": "0", "den": "1"},
        }
    )
    result = compute_circle_inversion(request)
    assert result.complete is True
