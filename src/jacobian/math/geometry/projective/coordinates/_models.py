"""Typed wire contracts for projective coordinate operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by the geometry contracts."""

    return PydanticCustomError(f"geometry.{reason}", message)


MAX_DIM = 16
MAX_PROJECTIVE_COORDINATE_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS // 2


class RationalProjectivePoint(StrictModel):
    """A projective point [x0 : x1 : ... : xn] over the rationals."""

    coordinates: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DIM + 1
    )

    @model_validator(mode="after")
    def require_not_all_zero(self) -> Self:
        if all(c.as_fraction() == 0 for c in self.coordinates):
            raise _validation_error(
                "projective_point_least_nonzero_coordinate",
                "projective point must have at least one nonzero coordinate",
            )
        return self


class RationalPointConstructRequest(StrictModel):
    coordinates: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_DIM + 1,
        description=(
            "Homogeneous coordinates whose rational components are at most "
            "16,384 digits so every normalization ratio remains representable."
        ),
    )


class StandardChartRequest(StrictModel):
    point: RationalProjectivePoint = Field(
        description=(
            "A rational projective point whose coordinate components are at most "
            "16,384 digits so every chart ratio remains representable."
        )
    )
    chart_index: int = Field(ge=0)


class ChartTransitionRequest(StrictModel):
    point: RationalProjectivePoint = Field(
        description=(
            "A rational projective point whose coordinate components are at most "
            "16,384 digits so every target-chart ratio remains representable."
        )
    )
    chart_i: int = Field(ge=0)
    chart_j: int = Field(ge=0)


# Results


class RationalPointConstructResult(StrictModel):
    point: RationalProjectivePoint
    scale: CanonicalRational


class StandardChartResult(StrictModel):
    affine_point: tuple[CanonicalRational, ...]
    chart_index: int = Field(ge=0)


class ChartTransitionResult(StrictModel):
    status: Literal["DEFINED", "OUTSIDE_TARGET_CHART"]
    transition: tuple[CanonicalRational, ...] | None
    chart_i: int = Field(ge=0)
    chart_j: int = Field(ge=0)
    projective_dimension: int = Field(ge=0, le=MAX_DIM)

    @model_validator(mode="after")
    def require_status_consistency(self) -> Self:
        if (self.status == "DEFINED") != (self.transition is not None):
            raise _validation_error(
                "defined_carry_target_chart_coordinates_outside",
                "DEFINED must carry target-chart coordinates and "
                "OUTSIDE_TARGET_CHART must not",
            )
        if (
            self.chart_i > self.projective_dimension
            or self.chart_j > self.projective_dimension
        ):
            raise _validation_error(
                "chart_axes_belong_projective_point",
                "chart axes must belong to the projective point",
            )
        if (
            self.transition is not None
            and len(self.transition) != self.projective_dimension
        ):
            raise _validation_error(
                "defined_transition_contain_every_target_chart",
                "defined transition must contain every target-chart coordinate",
            )
        return self
