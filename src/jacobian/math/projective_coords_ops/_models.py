"""Typed wire contracts for projective coordinate operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_DIM = 16


class RationalProjectivePoint(StrictModel):
    """A projective point [x0 : x1 : ... : xn] over the rationals."""

    coordinates: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DIM + 1
    )

    @model_validator(mode="after")
    def require_not_all_zero(self) -> Self:
        if all(c.as_fraction() == 0 for c in self.coordinates):
            raise ValueError(
                "projective point must have at least one nonzero coordinate"
            )
        return self


class RationalPointConstructRequest(StrictModel):
    coordinates: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DIM + 1
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if all(c.as_fraction() == 0 for c in self.coordinates):
            raise ValueError(
                "projective point must have at least one nonzero coordinate"
            )
        return self


class StandardChartRequest(StrictModel):
    point: RationalProjectivePoint
    chart_index: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if self.chart_index >= len(self.point.coordinates):
            raise ValueError("chart_index out of range")
        if self.point.coordinates[self.chart_index].as_fraction() == 0:
            raise ValueError("chart coordinate must be nonzero")
        return self


class ChartTransitionRequest(StrictModel):
    point: RationalProjectivePoint
    chart_i: int = Field(ge=0)
    chart_j: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        n = len(self.point.coordinates)
        if self.chart_i >= n or self.chart_j >= n:
            raise ValueError("chart index out of range")
        if self.point.coordinates[self.chart_i].as_fraction() == 0:
            raise ValueError("chart_i coordinate must be nonzero")
        if self.point.coordinates[self.chart_j].as_fraction() == 0:
            raise ValueError("chart_j coordinate must be nonzero")
        return self


# Results


class RationalPointConstructResult(StrictModel):
    canonical: tuple[CanonicalRational, ...]
    scale: CanonicalRational
    projective_dimension: int = Field(ge=0)
    method: str = "FIRST_NONZERO_SCALE"


class StandardChartResult(StrictModel):
    affine_point: tuple[CanonicalRational, ...]
    chart_index: int = Field(ge=0)
    method: str = "DEHOMOGENIZATION"


class ChartTransitionResult(StrictModel):
    transition: tuple[CanonicalRational, ...]
    chart_i: int = Field(ge=0)
    chart_j: int = Field(ge=0)
    method: str = "CHART_TRANSITION"
