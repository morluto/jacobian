"""Source-binding tests for circumradius profiles and pinned distances."""

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry._models import (
    CircumradiusProfileRequest,
    CircumradiusProfileResult,
    LabelledPoint2D,
    RationalPoint2D,
)
from jacobian.math.geometry._operations import circumradius_profile
from jacobian.math.geometry._pinned_distances import (
    PinnedDistanceRequest,
    PinnedDistanceResult,
    compute_pinned_distances,
)


def _pt(xn: str, xd: str, yn: str, yd: str) -> RationalPoint2D:
    return RationalPoint2D(
        x=CanonicalRational(num=xn, den=xd),
        y=CanonicalRational(num=yn, den=yd),
    )


def _square_points() -> tuple[LabelledPoint2D, ...]:
    return (
        LabelledPoint2D(label="A", point=_pt("0", "1", "0", "1")),
        LabelledPoint2D(label="B", point=_pt("1", "1", "0", "1")),
        LabelledPoint2D(label="C", point=_pt("1", "1", "1", "1")),
        LabelledPoint2D(label="D", point=_pt("0", "1", "1", "1")),
    )


class TestCircumradiusProfile:
    def test_unit_square_profile(self) -> None:
        request = CircumradiusProfileRequest(points=_square_points())
        result = circumradius_profile(request)
        assert result.point_count == 4
        assert result.triple_count == 4
        half = CanonicalRational.from_fraction(__import__("fractions").Fraction(1, 2))
        assert all(
            entry.squared_circumradius == half for entry in result.entries
        )

    def test_serialized_result_revalidates(self) -> None:
        result = circumradius_profile(
            CircumradiusProfileRequest(points=_square_points())
        )
        revived = CircumradiusProfileResult.model_validate(result.model_dump())
        assert revived == result

    def test_result_retains_source_points(self) -> None:
        points = _square_points()
        result = circumradius_profile(CircumradiusProfileRequest(points=points))
        assert result.points == points

    def test_tampered_radius_rejected(self) -> None:
        result = circumradius_profile(
            CircumradiusProfileRequest(points=_square_points())
        )
        tampered = list(result.entries)
        tampered[0] = tampered[0].model_copy(
            update={
                "squared_circumradius": CanonicalRational(
                    num="12345679", den="1"
                )
            }
        )
        with pytest.raises(
            ValidationError, match="squared circumradius must follow"
        ):
            CircumradiusProfileResult(
                point_count=result.point_count,
                points=result.points,
                triple_count=result.triple_count,
                entries=tuple(tampered),
            )

    def test_missing_source_points_rejected(self) -> None:
        result = circumradius_profile(
            CircumradiusProfileRequest(points=_square_points())
        )
        payload = result.model_dump()
        del payload["points"]
        with pytest.raises(ValidationError):
            CircumradiusProfileResult.model_validate(payload)

    def test_collinearity_flag_mismatch_rejected(self) -> None:
        collinear_points = (
            LabelledPoint2D(label="A", point=_pt("0", "1", "0", "1")),
            LabelledPoint2D(label="B", point=_pt("1", "1", "0", "1")),
            LabelledPoint2D(label="C", point=_pt("2", "1", "0", "1")),
            LabelledPoint2D(label="D", point=_pt("0", "1", "1", "1")),
        )
        result = circumradius_profile(
            CircumradiusProfileRequest(points=collinear_points)
        )
        flipped = []
        for entry in result.entries:
            if entry.indices == (0, 1, 2) and entry.collinear:
                flipped.append(
                    entry.model_copy(
                        update={
                            "collinear": False,
                            "squared_circumradius": CanonicalRational(
                                num="1", den="1"
                            ),
                        }
                    )
                )
            else:
                flipped.append(entry)
        with pytest.raises(
            ValidationError, match="collinearity must equal the source-point"
        ):
            CircumradiusProfileResult(
                point_count=result.point_count,
                points=result.points,
                triple_count=result.triple_count,
                entries=tuple(flipped),
            )

    def test_label_mismatch_rejected(self) -> None:
        result = circumradius_profile(
            CircumradiusProfileRequest(points=_square_points())
        )
        entries = list(result.entries)
        first = entries[0]
        other_label = next(
            label
            for label in ("A", "B", "C", "D")
            if label not in first.labels
        )
        entries[0] = first.model_copy(
            update={"labels": (other_label, *first.labels[1:])}
        )
        with pytest.raises(ValidationError, match="labels must match"):
            CircumradiusProfileResult(
                point_count=result.point_count,
                points=result.points,
                triple_count=result.triple_count,
                entries=tuple(entries),
            )

    def test_oversized_coordinates_rejected(self) -> None:
        """Six unrelated 4001-digit coordinate denominators compound to a
        squared circumradius beyond the canonical limit; admission must
        reject the configuration before execution."""

        def tall(i: int) -> str:
            return str(10**4000 + 2 * i + 1)

        coords = [(tall(0), tall(1)), (tall(2), tall(3)), (tall(4), tall(5))]
        points = tuple(
            LabelledPoint2D(
                label=label,
                point=RationalPoint2D(
                    x=CanonicalRational(num="1", den=xn),
                    y=CanonicalRational(num="1", den=xd),
                ),
            )
            for label, (xn, xd) in zip(("A", "B", "C"), coords, strict=True)
        )
        with pytest.raises(ValidationError):
            CircumradiusProfileRequest(points=points)


class TestPinnedDistances:
    def test_unit_square_profile(self) -> None:
        request = PinnedDistanceRequest(
            anchor=_pt("0", "1", "0", "1"),
            points=(
                _pt("0", "1", "0", "1"),
                _pt("1", "1", "0", "1"),
                _pt("1", "1", "1", "1"),
                _pt("0", "1", "1", "1"),
            ),
        )
        result = compute_pinned_distances(request)
        assert result.distinct_line_count == 6
        assert result.min_squared_distance is not None
        assert result.min_squared_distance.squared_distance_numerator == "0"
        total_pairs = sum(len(line.source_pairs) for line in result.lines)
        assert total_pairs == 6

    def test_serialized_result_revalidates(self) -> None:
        request = PinnedDistanceRequest(
            anchor=_pt("0", "1", "0", "1"),
            points=(
                _pt("-1", "1", "1", "1"),
                _pt("1", "1", "-1", "1"),
            ),
        )
        result = compute_pinned_distances(request)
        revived = PinnedDistanceResult.model_validate(result.model_dump())
        assert revived == result

    def test_fabricated_distance_rejected(self) -> None:
        anchor = _pt("0", "1", "0", "1")
        points = (_pt("-1", "1", "1", "1"), _pt("1", "1", "-1", "1"))
        with pytest.raises(
            ValidationError, match="lines must be the exact profile"
        ):
            PinnedDistanceResult(
                anchor=anchor,
                points=points,
                lines=(
                    {
                        "squared_distance_numerator": "123",
                        "squared_distance_denominator": "1",
                        "source_pairs": ((0, 1),),
                    },
                ),
                distinct_line_count=1,
                min_squared_distance={
                    "squared_distance_numerator": "123",
                    "squared_distance_denominator": "1",
                    "source_pairs": ((0, 1),),
                },
            )

    def test_out_of_range_pair_rejected(self) -> None:
        anchor = _pt("0", "1", "0", "1")
        points = (_pt("1", "1", "0", "1"), _pt("0", "1", "1", "1"))
        real = compute_pinned_distances(
            PinnedDistanceRequest(anchor=anchor, points=points)
        )
        lines = list(real.lines)
        lines[0] = lines[0].model_copy(update={"source_pairs": ((0, 7),)})
        with pytest.raises(
            ValidationError, match="lines must be the exact profile"
        ):
            PinnedDistanceResult(
                anchor=anchor,
                points=points,
                lines=tuple(lines),
                distinct_line_count=len(lines),
                min_squared_distance=real.min_squared_distance,
            )

    def test_dropped_pair_coverage_rejected(self) -> None:
        anchor = _pt("0", "1", "0", "1")
        points = (
            _pt("1", "1", "0", "1"),
            _pt("2", "1", "0", "1"),
            _pt("3", "1", "0", "1"),
        )
        real = compute_pinned_distances(
            PinnedDistanceRequest(anchor=anchor, points=points)
        )
        assert len(real.lines) == 1
        assert len(real.lines[0].source_pairs) == 3
        lines = []
        for line in real.lines:
            if len(line.source_pairs) > 1:
                line = line.model_copy(
                    update={"source_pairs": line.source_pairs[:1]}
                )
            lines.append(line)
        with pytest.raises(
            ValidationError, match="lines must be the exact profile"
        ):
            PinnedDistanceResult(
                anchor=anchor,
                points=points,
                lines=tuple(lines),
                distinct_line_count=len(lines),
                min_squared_distance=real.min_squared_distance,
            )

    def test_wrong_minimum_rejected(self) -> None:
        anchor = _pt("0", "1", "0", "1")
        points = (
            _pt("1", "1", "0", "1"),
            _pt("0", "1", "1", "1"),
            _pt("1", "1", "1", "1"),
        )
        real = compute_pinned_distances(
            PinnedDistanceRequest(anchor=anchor, points=points)
        )
        not_min = max(real.lines, key=lambda line: (
            int(line.squared_distance_numerator),
            -int(line.squared_distance_denominator),
        ))
        with pytest.raises(
            ValidationError, match="min_squared_distance must be the minimum"
        ):
            PinnedDistanceResult(
                anchor=anchor,
                points=points,
                lines=real.lines,
                distinct_line_count=real.distinct_line_count,
                min_squared_distance=not_min,
            )
