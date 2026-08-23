"""Retained-source admission for circumradius profile results."""

import pytest
from pydantic import ValidationError

from jacobian.math.geometry._models import (
    CircumradiusProfileRequest,
    CircumradiusProfileResult,
    CircumradiusTripleEntry,
    LabelledPoint2D,
)


def _point(label: str, num_x: str, num_y: str) -> LabelledPoint2D:
    return LabelledPoint2D(
        label=label,
        point={
            "x": {"num": num_x, "den": "1"},
            "y": {"num": num_y, "den": "1"},
        },
    )


def _collinear_triple_entry(labels: tuple[str, str, str]) -> CircumradiusTripleEntry:
    return CircumradiusTripleEntry(
        labels=labels,
        indices=(0, 1, 2),
        collinear=True,
        squared_circumradius=None,
    )


def test_collinear_profile_round_trip() -> None:
    points = (
        _point("A", "0", "0"),
        _point("B", "1", "0"),
        _point("C", "2", "0"),
    )
    request = CircumradiusProfileRequest(points=points)
    result = CircumradiusProfileResult(
        point_count=3,
        triple_count=1,
        entries=(_collinear_triple_entry(("A", "B", "C")),),
        points=request.points,
    )
    assert result.exactness == "EXACT_RATIONAL"


def test_duplicate_retained_sources_are_rejected() -> None:
    """A decoded profile cannot retain duplicate labels or coordinates that
    no accepted request could have supplied."""
    duplicated = (
        _point("A", "0", "0"),
        _point("A", "0", "0"),
        _point("C", "2", "0"),
    )
    entry = _collinear_triple_entry(("A", "A", "C"))
    with pytest.raises(ValidationError, match="unique"):
        CircumradiusProfileResult(
            point_count=3,
            triple_count=1,
            entries=(entry,),
            points=duplicated,
        )
    with pytest.raises(ValidationError, match="unique"):
        CircumradiusProfileRequest(points=duplicated)


def test_oversized_retained_coordinates_are_rejected() -> None:
    """Retained coordinates outside the request's height bound must fail
    before the exact replay performs unbounded arithmetic."""
    big = "1" + "0" * 1024
    points = (
        _point("A", "0", "0"),
        _point("B", big, "0"),
        _point("C", "2", "0"),
    )
    entry = _collinear_triple_entry(("A", "B", "C"))
    with pytest.raises(ValidationError, match="1024-digit"):
        CircumradiusProfileResult(
            point_count=3,
            triple_count=1,
            entries=(entry,),
            points=points,
        )
