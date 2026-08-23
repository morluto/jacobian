"""Contract tests for the exact circumradius profile operation."""

from fractions import Fraction

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


def _unit_right_triangle() -> CircumradiusProfileRequest:
    return CircumradiusProfileRequest(
        points=(
            _labelled("A", "0", "0"),
            _labelled("B", "1", "0"),
            _labelled("C", "0", "1"),
        )
    )


class TestCircumradiusProfileKnownAnswer:
    def test_unit_right_triangle_has_squared_circumradius_one_half(self) -> None:
        result = circumradius_profile(_unit_right_triangle())
        assert len(result.entries) == 1
        entry = result.entries[0]
        assert entry.collinear is False
        assert entry.squared_circumradius is not None
        assert entry.squared_circumradius.num == "1"
        assert entry.squared_circumradius.den == "2"
        assert entry.labels == ("A", "B", "C")
        assert entry.indices == (0, 1, 2)

    def test_square_profile_is_complete(self) -> None:
        request = CircumradiusProfileRequest(
            points=(
                _labelled("A", "0", "0"),
                _labelled("B", "1", "0"),
                _labelled("C", "1", "1"),
                _labelled("D", "0", "1"),
            )
        )
        result = circumradius_profile(request)
        assert result.point_count == 4
        assert result.triple_count == 4
        assert len(result.entries) == 4
        assert all(not entry.collinear for entry in result.entries)
        radii = {entry.squared_circumradius.as_fraction() for entry in result.entries}
        assert radii == {Fraction(1, 2)}

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

    def test_profile_exceeding_aggregate_transport_is_rejected(self) -> None:
        """24 near-collinear 302-digit points would emit ~10.9 MB of radii.

        Each coordinate pair (i, 1/(10^300+i)) is individually canonical, but
        the complete C(24,3)-entry profile cannot fit the canonical output
        limit, so admission must reject the configuration before any
        computation rather than fail during serialization.
        """
        big = 10**300
        points = tuple(
            LabelledPoint2D(
                label=f"P{i}",
                point=RationalPoint2D(
                    x={"num": str(i), "den": "1"},
                    y={"num": "1", "den": str(big + 2 * i + 1)},
                ),
            )
            for i in range(1, 25)
        )
        with pytest.raises(ValidationError, match="100-digit"):
            CircumradiusProfileRequest(points=points)

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
        result = circumradius_profile(_unit_right_triangle())
        payload = result.model_dump()
        payload["entries"][0]["squared_circumradius"] = {
            "num": "7",
            "den": "4",
        }
        with pytest.raises(ValidationError, match="does not match"):
            CircumradiusProfileResult.model_validate(payload)


class TestCollinearLabelBinding:
    def test_forged_labels_on_collinear_entry_rejected(self) -> None:
        request = CircumradiusProfileRequest(
            points=(
                _labelled("A", "0", "0"),
                _labelled("B", "1", "0"),
                _labelled("C", "2", "0"),
            )
        )
        result = circumradius_profile(request)
        assert result.entries[0].collinear is True
        payload = result.model_dump()
        payload["entries"][0]["labels"] = ["A", "B", "Z"]
        with pytest.raises(ValidationError):
            CircumradiusProfileResult.model_validate(payload)
