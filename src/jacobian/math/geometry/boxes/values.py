"""Canonical exact values for rational axis-aligned boxes."""

from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_CANONICAL_BOX_DIMENSION = 64


class RationalClosedInterval(StrictModel):
    """One nonempty closed rational interval, including a singleton."""

    lower: CanonicalRational
    upper: CanonicalRational

    @model_validator(mode="after")
    def require_ordered_endpoints(self) -> Self:
        if self.lower.as_fraction() > self.upper.as_fraction():
            raise ValueError("closed interval lower endpoint must not exceed upper")
        return self


class RationalAxisAlignedBox(StrictModel):
    """A closed rational box in standard ordered coordinates.

    ``intervals=null`` is the canonical empty box in the declared dimension.
    Otherwise the tuple contains exactly one closed interval for every standard
    coordinate axis; equal endpoints are valid and give a measure-zero box.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A rational axis-aligned box in standard coordinate order. "
                "Use intervals=null for the canonical empty box; otherwise "
                "provide exactly dimension closed intervals with lower <= upper."
            ),
            "examples": [
                {
                    "dimension": 2,
                    "intervals": [
                        {
                            "lower": {"num": "0", "den": "1"},
                            "upper": {"num": "1", "den": "1"},
                        },
                        {
                            "lower": {"num": "-1", "den": "1"},
                            "upper": {"num": "2", "den": "1"},
                        },
                    ],
                }
            ],
        }
    )

    dimension: StrictInt = Field(ge=1, le=MAX_CANONICAL_BOX_DIMENSION)
    intervals: tuple[RationalClosedInterval, ...] | None = Field(
        max_length=MAX_CANONICAL_BOX_DIMENSION,
        description=(
            "Null exactly for the empty box; otherwise one interval per "
            "standard coordinate axis in index order."
        ),
    )

    @model_validator(mode="after")
    def require_dimension(self) -> Self:
        if self.intervals is not None and len(self.intervals) != self.dimension:
            raise ValueError("box intervals must have length equal to dimension")
        return self

    @property
    def is_empty(self) -> bool:
        return self.intervals is None


__all__ = [
    "RationalAxisAlignedBox",
    "RationalClosedInterval",
]
