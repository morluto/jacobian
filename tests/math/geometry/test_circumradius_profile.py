"""Contract tests for the bounded exact circumradius profile operation."""

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError
from sympy import nextprime

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.math.geometry._models import (
    CircumradiusProfileRequest,
    CircumradiusProfileResult,
    LabelledPoint2D,
    RationalPoint2D,
)
from jacobian.math.geometry._operations import circumradius_profile


def _point(x: str, y: str) -> RationalPoint2D:
    return RationalPoint2D(
        x={"num": x, "den": "1"},
        y={"num": y, "den": "1"},
    )


def _labelled(index: int, x: str, y: str) -> LabelledPoint2D:
    return LabelledPoint2D(label=f"p{index}", point=_point(x, y))


class TestCircumradiusProfileKnownAnswer:
    def test_unit_square_triples_are_half_squared_radius(self) -> None:
        request = CircumradiusProfileRequest(
            points=(
                _labelled(0, "0", "0"),
                _labelled(1, "1", "0"),
                _labelled(2, "1", "1"),
                _labelled(3, "0", "1"),
            )
        )
        result = circumradius_profile(request)
        assert result.point_count == 4
        assert result.triple_count == 4
        assert len(result.entries) == 4
        for entry in result.entries:
            assert not entry.collinear
            assert entry.squared_circumradius is not None
            assert entry.squared_circumradius.as_fraction() == Fraction(1, 2)

    def test_collinear_triple_is_flagged_without_value(self) -> None:
        request = CircumradiusProfileRequest(
            points=(
                _labelled(0, "0", "0"),
                _labelled(1, "1", "0"),
                _labelled(2, "2", "0"),
            )
        )
        result = circumradius_profile(request)
        assert result.entries[0].collinear
        assert result.entries[0].squared_circumradius is None


class TestCircumradiusProfileAdmissionBound:
    def test_coordinates_at_the_64_digit_height_are_admitted(self) -> None:
        denominator = str(nextprime(10**63))
        points = (
            LabelledPoint2D(
                label="p0",
                point=RationalPoint2D(
                    x=CanonicalRational(num="9" * 64, den=denominator),
                    y=CanonicalRational(num="1", den="1"),
                ),
            ),
            LabelledPoint2D(
                label="p1",
                point=RationalPoint2D(
                    x=CanonicalRational(num="-" + "9" * 64, den="1"),
                    y=CanonicalRational(num="1", den="1"),
                ),
            ),
            LabelledPoint2D(
                label="p2",
                point=RationalPoint2D(
                    x=CanonicalRational(num="0", den="1"),
                    y=CanonicalRational(num="1", den="1"),
                ),
            ),
        )
        admitted = CircumradiusProfileRequest(points=points)
        assert len(admitted.points) == 3

    def test_coordinates_above_the_64_digit_height_are_rejected(self) -> None:
        denominator = str(nextprime(10**63))
        with pytest.raises(ValidationError, match="conservative"):
            CircumradiusProfileRequest(
                points=(
                    LabelledPoint2D(
                        label="p0",
                        point=RationalPoint2D(
                            x=CanonicalRational(num="1" * 65, den=denominator),
                            y=CanonicalRational(num="1", den="1"),
                        ),
                    ),
                    LabelledPoint2D(label="p1", point=_point("1", "0")),
                    LabelledPoint2D(label="p2", point=_point("0", "1")),
                )
            )

    def test_more_than_32_points_is_rejected(self) -> None:
        points = tuple(_labelled(i, str(i), str(i * i)) for i in range(33))
        with pytest.raises(ValidationError):
            CircumradiusProfileRequest(points=points)


class TestCircumradiusProfileAggregateTransportBound:
    def test_full_profile_at_the_admitted_bound_encodes_inside_transport(self) -> None:
        """The scaled-parabola family that overflowed transport stays bounded."""
        denominator = str(nextprime(10**63))
        labelled = tuple(
            LabelledPoint2D(
                label=f"p{i}",
                point=RationalPoint2D(
                    x=CanonicalRational(num=str(i), den=denominator),
                    y=CanonicalRational(num=str(i * i), den=denominator),
                ),
            )
            for i in range(1, 33)
        )
        request = CircumradiusProfileRequest(points=labelled)
        result = circumradius_profile(request)
        assert result.triple_count == 4960
        encoded = encode_strict_json(result.model_dump(mode="json"))
        assert len(encoded) < 10 * 1024 * 1024
        # The exact profile must round-trip through the result model's replays.
        CircumradiusProfileResult.model_validate(json.loads(encoded.decode()))

    def test_formerly_accepted_overflowing_configuration_is_rejected(self) -> None:
        """A 256-digit-height configuration no longer reaches execution."""
        denominator = str(nextprime(10**255))
        with pytest.raises(ValidationError, match="conservative"):
            CircumradiusProfileRequest(
                points=tuple(
                    LabelledPoint2D(
                        label=f"p{i}",
                        point=RationalPoint2D(
                            x=CanonicalRational(num=str(i), den=denominator),
                            y=CanonicalRational(num=str(i * i), den=denominator),
                        ),
                    )
                    for i in range(1, 5)
                )
            )
