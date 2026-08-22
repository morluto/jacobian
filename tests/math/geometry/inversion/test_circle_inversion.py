"""Tests for circle inversion of rational planar points."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
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


def _point(x: int, y: int):
    from jacobian.math.geometry._models import RationalPoint2D

    return RationalPoint2D.model_validate(
        {
            "x": {"num": str(x), "den": "1"},
            "y": {"num": str(y), "den": "1"},
        }
    )


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
                center=_point(0, 0),
                power={"num": "0", "den": "1"},
                point=_point(1, 0),
            )


class TestWireAdapter:
    def test_compute_circle_inversion(self) -> None:
        request = CircleInversionRequest(
            center=_point(0, 0),
            power={"num": "1", "den": "1"},
            point=_point(4, 0),
        )
        result = compute_circle_inversion(request)
        assert result.inverted_point.x.num == "1"
        assert result.inverted_point.x.den == "4"
        assert result.inverted_point.y.num == "0"
        assert result.inverted_point.y.den == "1"

    def test_non_origin_center(self) -> None:
        request = CircleInversionRequest(
            center=_point(3, 2),
            power={"num": "5", "den": "1"},
            point=_point(1, 7),
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


class TestSourceBindingAndAdmission:
    def test_result_composes_as_next_request_point(self) -> None:
        """The serialized inverted point feeds the next call unchanged."""
        request = CircleInversionRequest(
            center=_point(0, 0),
            power={"num": "1", "den": "1"},
            point=_point(4, 2),
        )
        result = compute_circle_inversion(request)
        follow_up = CircleInversionRequest.model_validate(
            {
                "center": result.inverted_point.model_dump(),
                "power": {"num": "1", "den": "1"},
                "point": result.point.model_dump(),
            }
        )
        assert follow_up.point == result.point

    def test_multiplicative_growth_rejected_before_execution(self) -> None:
        """Six independent ~7,900-digit components overflow the result."""
        big = "1" + "0" * 7900
        with pytest.raises(ValidationError, match="canonical"):
            CircleInversionRequest(
                center=_point(0, 0),
                power=CanonicalRational(num=big, den="3"),
                point=RationalPoint2D(
                    x=CanonicalRational(num="1", den=big),
                    y=CanonicalRational(num="3", den=big),
                ),
            )

    def test_moderate_components_still_admitted(self) -> None:
        big = "1" + "0" * 100
        request = CircleInversionRequest(
            center=_point(0, 0),
            power=CanonicalRational(num=big, den="3"),
            point=RationalPoint2D(
                x=CanonicalRational(num="1", den=big),
                y=CanonicalRational(num="3", den=big),
            ),
        )
        assert compute_circle_inversion(request).inverted_point is not None
