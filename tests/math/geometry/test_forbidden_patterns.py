"""Admission bounds and known-answer checks for forbidden-pattern screening."""

from pydantic import ValidationError
import pytest

from jacobian.math.geometry._models import (
    ForbiddenConfiguration,
    ForbiddenLabelledPoint,
)
from jacobian.math.geometry._profiles import PROFILE_OPERATIONS


def _point(x: str, y: str) -> ForbiddenLabelledPoint:
    return ForbiddenLabelledPoint(
        label=x + "_" + y,
        point={"x": {"num": x, "den": "1"}, "y": {"num": y, "den": "1"}},
    )


def test_configuration_beyond_the_twenty_four_point_bound_is_rejected():
    points = tuple(
        _point(str(i), str(i * i % 7))
        for i in range(25)
    )
    assert len(points) == 25
    with pytest.raises(ValidationError):
        ForbiddenConfiguration(points=points)


def test_oversized_coordinate_height_is_rejected_at_admission():
    big = str(10**2049)
    with pytest.raises(ValidationError, match="coordinate"):
        ForbiddenConfiguration(
            points=(
                ForbiddenLabelledPoint(
                    label="o",
                    point={
                        "x": {"num": "0", "den": "1"},
                        "y": {"num": big, "den": "1"},
                    },
                ),
                _point("1", "0"),
                _point("2", "1"),
            )
        )


def test_tool_declares_the_screening_operation():
    ids = {tool.operation_id for tool in PROFILE_OPERATIONS}
    assert "geometry.configuration.forbidden_patterns.check" in ids
