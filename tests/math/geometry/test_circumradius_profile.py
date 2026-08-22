"""Contract tests for the bounded circumradius profile operation."""

import pytest
from pydantic import ValidationError

from jacobian.math.geometry._models import (
    CircumradiusProfileRequest,
    CircumradiusProfileResult,
    LabelledPoint2D,
    RationalPoint2D,
)
from jacobian.math.geometry._operations import circumradius_profile


def _point(x: str, y: str) -> RationalPoint2D:
    return RationalPoint2D(x={"num": x, "den": "1"}, y={"num": y, "den": "1"})


def _labelled(label: str, x: str, y: str) -> LabelledPoint2D:
    return LabelledPoint2D(label=label, point=_point(x, y))


def _right_triangle() -> list[LabelledPoint2D]:
    return [
        _labelled("A", "0", "0"),
        _labelled("B", "1", "0"),
        _labelled("C", "0", "1"),
    ]


class TestAdmissionBounds:
    def test_coordinates_beyond_1024_digits_rejected(self) -> None:
        huge = "1" + "0" * 1024
        with pytest.raises(ValidationError, match="1024-digit"):
            CircumradiusProfileRequest(
                points=[
                    _labelled("A", huge, "0"),
                    _labelled("B", "1", "0"),
                    _labelled("C", "0", "1"),
                ]
            )

    def test_more_than_24_points_rejected(self) -> None:
        points = [_labelled(f"P{i}", str(i), str(i * i % 13)) for i in range(25)]
        with pytest.raises(ValidationError):
            CircumradiusProfileRequest(points=points)


class TestSourceBoundResult:
    def test_result_retains_and_replays_configuration(self) -> None:
        request = CircumradiusProfileRequest(points=_right_triangle())
        result = circumradius_profile(request)
        assert result.points == request.points
        revalidated = CircumradiusProfileResult.model_validate(result.model_dump())
        assert revalidated.entries == result.entries

    def test_derived_triple_count(self) -> None:
        request = CircumradiusProfileRequest(points=_right_triangle())
        result = circumradius_profile(request)
        assert result.triple_count == 1

    def test_forged_entry_rejected(self) -> None:
        payload = circumradius_profile(
            CircumradiusProfileRequest(points=_right_triangle())
        ).model_dump()
        payload["entries"][0]["squared_circumradius"] = {"num": "7", "den": "3"}
        with pytest.raises(ValidationError, match="exact profile"):
            CircumradiusProfileResult.model_validate(payload)

    def test_wrong_point_count_rejected(self) -> None:
        payload = circumradius_profile(
            CircumradiusProfileRequest(points=_right_triangle())
        ).model_dump()
        payload["point_count"] = 4
        payload["triple_count"] = 4
        with pytest.raises(ValueError, match="point count"):
            CircumradiusProfileResult.model_validate(payload)

    def test_incomplete_coverage_rejected(self) -> None:
        from jacobian.math.geometry._models import CircumradiusTripleEntry

        four_points = [*_right_triangle(), _labelled("D", "2", "2")]

        def triple(a: int, b: int, c: int) -> CircumradiusTripleEntry:
            return CircumradiusTripleEntry(
                labels=(
                    four_points[a].label,
                    four_points[b].label,
                    four_points[c].label,
                ),
                indices=(a, b, c),
                collinear=True,
            )

        entries = [triple(0, 1, 2), triple(0, 1, 3), triple(0, 2, 3), triple(0, 1, 2)]
        with pytest.raises(ValueError, match="exactly once"):
            CircumradiusProfileResult(
                points=four_points,
                point_count=4,
                triple_count=4,
                entries=entries,
            )
