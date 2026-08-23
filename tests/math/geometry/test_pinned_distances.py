"""Contract tests for pinned-distance admission and source binding."""

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry._models import RationalPoint2D
from jacobian.math.geometry._pinned_distances import (
    PinnedDistanceRequest,
    PinnedDistanceResult,
    compute_pinned_distances,
)


def _pt(x: str, y: str) -> RationalPoint2D:
    return RationalPoint2D(
        x=CanonicalRational(num=x, den="1"),
        y=CanonicalRational(num=y, den="1"),
    )


def _request() -> PinnedDistanceRequest:
    return PinnedDistanceRequest(
        anchor=_pt("0", "0"),
        points=(_pt("0", "0"), _pt("3", "0"), _pt("0", "4")),
    )


class TestKnownAnswer:
    def test_profile(self):
        result = compute_pinned_distances(_request())
        assert result.distinct_line_count == len(result.lines) == 3
        assert result.min_squared_distance is not None


class TestResultAdmissionBounds:
    def test_oversized_result_point_set_rejected_before_replay(self):
        points = tuple(_pt(str(i), str(i * i)) for i in range(40))
        with pytest.raises(ValidationError, match="enumeration budget"):
            PinnedDistanceResult(
                anchor=_pt("0", "0"),
                points=points,
                lines=(),
                distinct_line_count=0,
                min_squared_distance=None,
            )

    def test_duplicate_points_in_result_rejected(self):
        with pytest.raises(ValidationError, match="unique"):
            PinnedDistanceResult(
                anchor=_pt("0", "0"),
                points=(_pt("1", "0"), _pt("1", "0")),
                lines=(),
                distinct_line_count=0,
                min_squared_distance=None,
            )
