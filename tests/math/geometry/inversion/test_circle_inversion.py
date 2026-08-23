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


def _point(x: str, y: str) -> dict:
    return {"x": {"num": x, "den": "1"}, "y": {"num": y, "den": "1"}}


def _rational(num: str, den: str = "1") -> dict:
    return {"num": num, "den": den}


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
                center=_point("0", "0"),
                power=_rational("0"),
                point=_point("1", "0"),
            )

    def test_inverting_the_center_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="center cannot be inverted"):
            CircleInversionRequest(
                center=_point("3", "-2"),
                power=_rational("1"),
                point=_point("3", "-2"),
            )


class TestClosedDomain:
    """Every accepted request must admit its own exact inverted point back."""

    def test_near_origin_point_whose_inverse_leaves_the_domain_is_rejected(
        self,
    ) -> None:
        # p = (1/10^6000, 1/(10^6000+1)) is itself inside the admission
        # bound, but its exact inverse has an ~18001-digit numerator, so the
        # advertised involution must refuse the request at admission instead
        # of returning a value that cannot re-enter the operation.
        with pytest.raises(ValidationError, match="closed inversion domain"):
            CircleInversionRequest(
                center=_point("0", "0"),
                power=_rational("1"),
                point={
                    "x": {"num": "1", "den": "1" + "0" * 6000},
                    "y": {"num": "1", "den": "9" * 6000 + "1"},
                },
            )

    def test_accepted_results_feed_back_unchanged(self) -> None:
        request = CircleInversionRequest(
            center=_point("3", "2"),
            power=_rational("5"),
            point=_point("1", "7"),
        )
        result = compute_circle_inversion(request)
        replay = CircleInversionRequest(
            center=result.center, power=result.power, point=result.inverted_point
        )
        round_trip = compute_circle_inversion(replay)
        assert round_trip.inverted_point == request.point


class TestWireAdapter:
    def test_compute_circle_inversion(self) -> None:
        request = CircleInversionRequest(
            center=_point("0", "0"),
            power=_rational("1"),
            point=_point("4", "0"),
        )
        result = compute_circle_inversion(request)
        assert isinstance(result.inverted_point, RationalPoint2D)
        assert result.inverted_point.x.num == "1"
        assert result.inverted_point.x.den == "4"
        assert result.inverted_point.y.num == "0"
        assert result.inverted_point.y.den == "1"

    def test_non_origin_center(self) -> None:
        request = CircleInversionRequest(
            center=_point("3", "2"),
            power=_rational("5"),
            point=_point("1", "7"),
        )
        result = compute_circle_inversion(request)
        assert isinstance(result, CircleInversionResult)
        # q = c + s(p-c)/||p-c||^2 = (3,2) + (5/29)(-2,5) = (77/29, 83/29).
        assert result.inverted_point == RationalPoint2D(
            x={"num": "77", "den": "29"}, y={"num": "83", "den": "29"}
        )

    def test_result_binds_the_exact_inversion(self) -> None:
        request = CircleInversionRequest(
            center=_point("3", "2"),
            power=_rational("5"),
            point=_point("1", "7"),
        )
        result = compute_circle_inversion(request)
        payload = result.model_dump()
        payload["inverted_point"]["x"]["num"] = "2"
        with pytest.raises(ValidationError, match="exact inversion result"):
            CircleInversionResult.model_validate(payload)


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

    def test_schema_uses_the_geometry_canonical_point_value(self) -> None:
        schema = CircleInversionRequest.model_json_schema()
        definitions = schema.get("$defs", {})
        assert "RationalPoint2D" in definitions
        assert set(schema["properties"]) == {"center", "power", "point"}
