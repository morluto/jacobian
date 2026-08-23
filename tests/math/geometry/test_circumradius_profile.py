"""Contract tests for the bounded exact circumradius profile operation."""

import pytest
from pydantic import ValidationError

from jacobian.math.geometry._models import (
    CircumradiusProfileRequest,
    CircumradiusProfileResult,
    LabelledPoint2D,
    RationalPoint2D,
)
from jacobian.math.geometry._operations import circumradius_profile


def _labelled(label: str, x: str, y: str) -> LabelledPoint2D:
    return LabelledPoint2D(
        label=label,
        point=RationalPoint2D(
            x={"num": x, "den": "1"},
            y={"num": y, "den": "1"},
        ),
    )


class TestCircumradiusProfileKnownAnswer:
    def test_unit_right_triangle_has_squared_circumradius_one_half(self) -> None:
        request = CircumradiusProfileRequest(
            points=(
                _labelled("A", "0", "0"),
                _labelled("B", "1", "0"),
                _labelled("C", "0", "1"),
            )
        )
        result = circumradius_profile(request)
        assert result.triple_count == 1
        entry = result.entries[0]
        assert entry.collinear is False
        assert entry.squared_circumradius is not None
        assert entry.squared_circumradius.num == "1"
        assert entry.squared_circumradius.den == "2"

    def test_collinear_triple_is_flagged_degenerate(self) -> None:
        request = CircumradiusProfileRequest(
            points=(
                _labelled("A", "0", "0"),
                _labelled("B", "1", "0"),
                _labelled("C", "2", "0"),
            )
        )
        result = circumradius_profile(request)
        assert result.entries[0].collinear is True
        assert result.entries[0].squared_circumradius is None


class TestCircumradiusAdmissionAndBinding:
    def test_extreme_coordinate_height_is_rejected_at_admission(self) -> None:
        """(0, 10^20000)-style configurations must fail before execution.

        The exact squared circumradius of that accepted triangle would be
        (10^40000+1)/4, whose numerator exceeds the canonical 32,768-digit
        rational limit and raised a result-model ValidationError after the
        request was accepted.
        """
        with pytest.raises(ValidationError, match="100-digit"):
            CircumradiusProfileRequest(
                points=(
                    _labelled("A", "0", "0"),
                    _labelled("B", "1", "0"),
                    _labelled("C", "0", "1" + "0" * 20000),
                )
            )

    def test_coordinates_beyond_100_digits_are_rejected(self) -> None:
        huge = "1" + "0" * 101
        with pytest.raises(ValidationError, match="100-digit"):
            CircumradiusProfileRequest(
                points=(
                    _labelled("A", huge, "0"),
                    _labelled("B", "1", "0"),
                    _labelled("C", "0", "1"),
                )
            )

    def test_largest_admitted_coordinates_round_trip(self) -> None:
        """A 100-digit configuration computes and revalidates its profile."""
        tall = str(10**99 - 7)
        request = CircumradiusProfileRequest(
            points=(
                _labelled("A", "0", "0"),
                _labelled("B", "1", "0"),
                _labelled("C", tall, "1"),
            )
        )
        result = circumradius_profile(request)
        assert result.entries[0].squared_circumradius is not None
        CircumradiusProfileResult.model_validate(result.model_dump(mode="json"))

    def test_forged_radius_is_rejected_by_source_replay(self) -> None:
        request = CircumradiusProfileRequest(
            points=(
                _labelled("A", "0", "0"),
                _labelled("B", "1", "0"),
                _labelled("C", "0", "1"),
            )
        )
        result = circumradius_profile(request)
        payload = result.model_dump()
        payload["entries"][0]["squared_circumradius"] = {"num": "7", "den": "4"}
        with pytest.raises(ValidationError):
            CircumradiusProfileResult.model_validate(payload)
