"""Tests for circle inversion of rational planar points."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry._models import RationalPoint2D
from jacobian.math.geometry.inversion._models import CircleInversionRequest
from jacobian.math.geometry.inversion._operations import compute_circle_inversion
from jacobian.math.geometry.inversion._tools import TOOLS


def _point(x: int, y: int) -> RationalPoint2D:
    return RationalPoint2D(
        x=CanonicalRational(num=str(x), den="1"),
        y=CanonicalRational(num=str(y), den="1"),
    )


class TestCircleInversion:
    def test_known_answer(self) -> None:
        result = compute_circle_inversion(
            CircleInversionRequest(
                center=_point(0, 0),
                power=CanonicalRational(num="1", den="1"),
                point=_point(4, 0),
            )
        )
        assert result.inverted_point.x.num == "1"
        assert result.inverted_point.x.den == "4"
        assert result.inverted_point.y.num == "0"

    def test_center_rejected(self) -> None:
        with pytest.raises(ValidationError, match="center"):
            CircleInversionRequest(
                center=_point(1, 1),
                power=CanonicalRational(num="1", den="1"),
                point=_point(1, 1),
            )

    def test_zero_power_rejected(self) -> None:
        with pytest.raises(ValidationError, match="positive"):
            CircleInversionRequest(
                center=_point(0, 0),
                power=CanonicalRational(num="0", den="1"),
                point=_point(1, 0),
            )

    def test_multiplicative_growth_rejected_before_execution(self) -> None:
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

    def test_result_composes_as_next_request_point(self) -> None:
        result = compute_circle_inversion(
            CircleInversionRequest(
                center=_point(0, 0),
                power=CanonicalRational(num="1", den="1"),
                point=_point(4, 2),
            )
        )
        follow_up = CircleInversionRequest.model_validate(
            {
                "center": result.inverted_point.model_dump(),
                "power": {"num": "1", "den": "1"},
                "point": result.point.model_dump(),
            }
        )
        assert follow_up.point == result.point

    def test_example_runs(self) -> None:
        for tool in TOOLS:
            for ex in tool.examples:
                request = tool.request_type.model_validate(ex.input)
                assert tool.run(request) is not None
