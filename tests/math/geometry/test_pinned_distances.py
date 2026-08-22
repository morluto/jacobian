"""Contract tests for pinned-distance minimum binding."""

from __future__ import annotations

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


class TestMinimumEntryBinding:
    def test_detached_minimum_with_foreign_pairs_rejected(self) -> None:
        genuine = compute_pinned_distances(_request())
        minimum = genuine.min_squared_distance
        assert minimum is not None
        detached = minimum.model_copy(
            update={"source_pairs": ((8, 9),)},
        )
        with pytest.raises(ValidationError, match="actual minimum line entry"):
            PinnedDistanceResult(
                anchor=genuine.anchor,
                points=genuine.points,
                lines=genuine.lines,
                distinct_line_count=genuine.distinct_line_count,
                min_squared_distance=detached,
            )

    def test_non_minimum_entry_rejected(self) -> None:
        genuine = compute_pinned_distances(_request())
        other = next(e for e in genuine.lines if e != genuine.min_squared_distance)
        with pytest.raises(ValidationError, match="actual minimum line entry"):
            PinnedDistanceResult(
                anchor=genuine.anchor,
                points=genuine.points,
                lines=genuine.lines,
                distinct_line_count=genuine.distinct_line_count,
                min_squared_distance=other,
            )
