"""Contract tests for the exact circumradius profile operation."""

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


def _labelled(label: str, x: str, y: str) -> LabelledPoint2D:
    return LabelledPoint2D(
        label=label,
        point=RationalPoint2D(
            x=CanonicalRational(num=x, den="1"),
            y=CanonicalRational(num=y, den="1"),
        ),
    )


class TestCircumradiusProfileKnownAnswer:
    def test_unit_right_triangle(self) -> None:
        request = CircumradiusProfileRequest(
            points=(
                _labelled("A", "0", "0"),
                _labelled("B", "1", "0"),
                _labelled("C", "0", "1"),
            )
        )
        result = circumradius_profile(request)
        assert len(result.entries) == 1
        assert result.entries[0].squared_circumradius is not None
        assert result.entries[0].squared_circumradius.num == "1"
        assert result.entries[0].squared_circumradius.den == "2"


class TestResultConfigurationAdmission:
    def _genuine(self) -> CircumradiusProfileResult:
        return circumradius_profile(
            CircumradiusProfileRequest(
                points=(
                    _labelled("A", "0", "0"),
                    _labelled("B", "1", "0"),
                    _labelled("C", "0", "1"),
                )
            )
        )

    def test_duplicate_coordinates_in_result_rejected(self) -> None:
        payload = self._genuine().model_dump()
        payload["points"][2]["point"] = {
            "x": {"num": "0", "den": "1"},
            "y": {"num": "0", "den": "1"},
        }
        with pytest.raises(ValidationError, match="unique"):
            CircumradiusProfileResult.model_validate(payload)

    def test_oversized_coordinates_in_result_rejected(self) -> None:
        big = "1" + "0" * 900
        genuine = self._genuine()
        payload = genuine.model_dump()
        points = list(payload["points"])
        points[2] = {
            "label": "C",
            "point": {
                "x": {"num": "0", "den": big},
                "y": {"num": "1", "den": big},
            },
        }
        payload["points"] = points
        # The replayed radius for the forged coordinates would also differ;
        # admission must reject the height before any of that work.
        with pytest.raises(ValidationError):
            CircumradiusProfileResult.model_validate(payload)
