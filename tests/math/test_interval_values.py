"""Canonical exact rational interval and box values."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math import intervals
from jacobian.math.intervals import ClosedRationalInterval, RationalBox


def _q(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(numerator, denominator)


def test_zero_dimensional_box_round_trips_with_its_parent_and_axis() -> None:
    box = RationalBox(variables=(), intervals=())

    assert RationalBox.model_validate_json(box.model_dump_json(), strict=True) == box
    assert box.domain == "QQ"
    assert box.variables == ()
    assert intervals.RationalBox is RationalBox


def test_box_preserves_ordered_axes_and_zero_width_coordinates() -> None:
    point = ClosedRationalInterval(lower=_q(2, 3), upper=_q(2, 3))
    span = ClosedRationalInterval(lower=_q(-1), upper=_q(4))
    box = RationalBox(variables=("y", "x"), intervals=(point, span))

    assert box.variables == ("y", "x")
    assert box.intervals == (point, span)


@pytest.mark.parametrize(
    ("payload", "error_type"),
    (
        (
            {
                "variables": ["x", "x"],
                "intervals": [
                    {
                        "lower": {"num": "0", "den": "1"},
                        "upper": {"num": "1", "den": "1"},
                    },
                    {
                        "lower": {"num": "0", "den": "1"},
                        "upper": {"num": "1", "den": "1"},
                    },
                ],
            },
            "interval.duplicate_variable",
        ),
        (
            {
                "variables": ["x"],
                "intervals": [],
            },
            "interval.axis_length",
        ),
        (
            {
                "variables": ["x"],
                "intervals": [
                    {
                        "lower": {"num": "2", "den": "1"},
                        "upper": {"num": "1", "den": "1"},
                    },
                ],
            },
            "interval.endpoint_order",
        ),
    ),
)
def test_box_rejects_noncanonical_axes_and_intervals(
    payload: dict[str, object],
    error_type: str,
) -> None:
    with pytest.raises(ValidationError) as error:
        RationalBox.model_validate(payload)
    assert error.value.errors()[0]["type"] == error_type
