"""Canonical exact rational intervals and axis-aligned boxes."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_RATIONAL_BOX_VARIABLES = 8


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"interval.{reason}", message)


IntervalVariable = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$", strict=True),
]


class ClosedRationalInterval(StrictModel):
    """One closed exact rational interval whose lower endpoint is at most its upper."""

    lower: CanonicalRational
    upper: CanonicalRational

    @model_validator(mode="after")
    def require_ordered_endpoints(self) -> Self:
        if self.lower.as_fraction() > self.upper.as_fraction():
            raise _validation_error(
                "endpoint_order",
                "interval lower endpoint must not exceed its upper endpoint",
            )
        return self


class RationalBox(StrictModel):
    """A closed rational box with one interval on each distinct ordered axis."""

    domain: Literal["QQ"] = "QQ"
    variables: tuple[IntervalVariable, ...] = Field(
        max_length=MAX_RATIONAL_BOX_VARIABLES,
        description=(
            "Complete ordered coordinate axis. Its order is part of the box's "
            "identity; the empty axis denotes the unique zero-dimensional box."
        ),
    )
    intervals: tuple[ClosedRationalInterval, ...] = Field(
        max_length=MAX_RATIONAL_BOX_VARIABLES,
        description=(
            "Closed coordinate intervals in the same order as variables; "
            "zero-width intervals are preserved."
        ),
    )

    @model_validator(mode="after")
    def require_complete_unique_axis(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error(
                "duplicate_variable", "rational-box variables must be unique"
            )
        if len(self.variables) != len(self.intervals):
            raise _validation_error(
                "axis_length",
                "rational-box variables and intervals must have the same length",
            )
        return self


__all__ = [
    "MAX_RATIONAL_BOX_VARIABLES",
    "ClosedRationalInterval",
    "IntervalVariable",
    "RationalBox",
]
