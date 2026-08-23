"""Canonical exact rational intervals and variable-bound boxes."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialVariable,
)


class ClosedRationalInterval(StrictModel):
    """One closed interval with exact rational endpoints."""

    lower: CanonicalRational
    upper: CanonicalRational

    @model_validator(mode="after")
    def require_ordered_endpoints(self) -> Self:
        if self.lower.as_fraction() > self.upper.as_fraction():
            raise ValueError(
                "interval lower endpoint must not exceed its upper endpoint"
            )
        return self


class RationalBox(StrictModel):
    """A closed rational box on one explicit ordered polynomial axis."""

    rational_box_schema_version: Literal["1"] = "1"
    domain: Literal["QQ"] = "QQ"
    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=1,
        max_length=MAX_POLYNOMIAL_VARIABLES,
        description=(
            "Complete ordered coordinate axis. Its order is part of the box's "
            "identity and must match a consumed polynomial exactly."
        ),
    )
    intervals: tuple[ClosedRationalInterval, ...] = Field(
        min_length=1,
        max_length=MAX_POLYNOMIAL_VARIABLES,
        description=(
            "Closed coordinate intervals in the same order as variables; "
            "zero-width intervals are preserved."
        ),
    )

    @model_validator(mode="after")
    def require_complete_unique_axis(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("rational-box variables must be unique")
        if len(self.variables) != len(self.intervals):
            raise ValueError(
                "rational-box variables and intervals must have the same length"
            )
        return self


__all__ = ["ClosedRationalInterval", "RationalBox"]
