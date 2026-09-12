"""Typed contracts for bounded regular plane-curve arclength enclosures.

The first admitted envelope is deliberately small: a nonempty regular affine
quadratic whose quadratic part is diagonal and positive (an axis-aligned
ellipse), or an exact empty constant source.  The result is still the
one-dimensional Hausdorff length of the zero set in the declared box; the
quadratic restriction is an admission boundary, not a change of meaning.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.analysis.intervals import RationalBox
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    require_polynomial_budget,
)

MAX_ARCLENGTH_SOURCE_TERMS = 6
MAX_ARCLENGTH_DEGREE = 4
MAX_ARCLENGTH_COEFFICIENT_DIGITS = 64
MAX_ARCLENGTH_BOX_ENDPOINT_DIGITS = 64
MAX_ARCLENGTH_PRECISION_BITS = 4096
MAX_ARCLENGTH_SEGMENTS = 1024
MAX_ARCLENGTH_WALL_SECONDS = 120

def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"plane_curve.arclength.{reason}", message)


class PlaneCurveArclengthBudget(StrictModel):
    """Request-scoped bounds charged before any Arb evaluation."""

    precision_bits: StrictInt = Field(default=192, ge=32, le=MAX_ARCLENGTH_PRECISION_BITS)
    max_segments: StrictInt = Field(default=128, ge=1, le=MAX_ARCLENGTH_SEGMENTS)
    wall_seconds: StrictInt = Field(default=60, ge=1, le=MAX_ARCLENGTH_WALL_SECONDS)


class PlaneCurveArclengthRequest(StrictModel):
    """Enclose ``H¹({F=0} ∩ box)`` for one bounded regular curve slice."""

    polynomial: RationalPolynomial
    box: RationalBox
    target_width: CanonicalRational = Field(
        description="Positive rational width for the returned lower/upper enclosure."
    )
    resource_budget: PlaneCurveArclengthBudget = Field(default_factory=PlaneCurveArclengthBudget)

    @model_validator(mode="before")
    @classmethod
    def preserve_raw_containers(cls, value: object) -> object:
        return canonicalize_json_containers(value)

    @model_validator(mode="after")
    def require_admitted_shape(self) -> Self:
        try:
            require_polynomial_budget(
                self.polynomial,
                maximum_terms=MAX_ARCLENGTH_SOURCE_TERMS,
                maximum_exponent=MAX_ARCLENGTH_DEGREE,
                maximum_coefficient_digits=MAX_ARCLENGTH_COEFFICIENT_DIGITS,
                label="arclength source polynomial",
            )
            require_bounded_rational(
                self.target_width,
                max_digits=MAX_ARCLENGTH_COEFFICIENT_DIGITS,
                label="arclength target width",
            )
            for interval in self.box.intervals:
                for endpoint in (interval.lower, interval.upper):
                    require_bounded_rational(
                        endpoint,
                        max_digits=MAX_ARCLENGTH_BOX_ENDPOINT_DIGITS,
                        label="arclength box endpoint",
                    )
        except ValueError as exc:
            raise _validation_error("source_bound", str(exc)) from exc
        if len(self.polynomial.variables) != 2 or self.box.variables != self.polynomial.variables:
            raise _validation_error(
                "axis",
                "the curve and box must use the same complete ordered two-variable axis",
            )
        if len(self.box.intervals) != 2:
            raise _validation_error("box", "a plane-curve box requires exactly two intervals")
        if self.target_width.as_fraction() <= 0:
            raise _validation_error("target_width", "target width must be positive")
        return self


class ArclengthSegment(StrictModel):
    """One source-bound rational parameter interval in the retained partition."""

    lower: CanonicalRational | None = None
    upper: CanonicalRational | None = None
    contribution_lower: CanonicalRational
    contribution_upper: CanonicalRational

    @model_validator(mode="after")
    def require_ordered_segment(self) -> Self:
        if self.lower is not None and self.upper is not None and self.lower.as_fraction() >= self.upper.as_fraction():
            raise _validation_error("segment_order", "arclength segment endpoints must be strictly ordered")
        if self.contribution_lower.as_fraction() < 0 or self.contribution_upper.as_fraction() < self.contribution_lower.as_fraction():
            raise _validation_error("segment_contribution", "segment contribution must be nonnegative and ordered")
        return self


class ArclengthEnclosed(StrictModel):
    status: Literal["ENCLOSED"] = "ENCLOSED"
    lower: CanonicalRational
    upper: CanonicalRational
    segments: tuple[ArclengthSegment, ...] = Field(min_length=1, max_length=MAX_ARCLENGTH_SEGMENTS)

    @model_validator(mode="after")
    def require_enclosure(self) -> Self:
        if self.lower.as_fraction() < 0 or self.upper.as_fraction() < self.lower.as_fraction():
            raise _validation_error("enclosure_order", "arclength enclosure must be nonnegative and ordered")
        if sum((segment.contribution_lower.as_fraction() for segment in self.segments), start=0) > self.lower.as_fraction():
            raise _validation_error("lower_reconstruction", "segment lower contributions exceed the aggregate lower bound")
        if sum((segment.contribution_upper.as_fraction() for segment in self.segments), start=0) < self.upper.as_fraction():
            raise _validation_error("upper_reconstruction", "segment upper contributions do not reach the aggregate upper bound")
        return self


class ArclengthEmpty(StrictModel):
    status: Literal["EMPTY"] = "EMPTY"
    lower: Literal[0] = 0
    upper: Literal[0] = 0
    segments: tuple[ArclengthSegment, ...] = ()


class ArclengthSingularUnsupported(StrictModel):
    status: Literal["SINGULAR_CASE_UNSUPPORTED"] = "SINGULAR_CASE_UNSUPPORTED"
    reason: Literal["SINGULAR_LEVEL_SET", "BOUNDARY_NONTRANSVERSE", "DEGENERATE_SOURCE"]


class ArclengthUnknown(StrictModel):
    status: Literal["UNKNOWN"] = "UNKNOWN"
    reason: Literal["BACKEND_UNAVAILABLE", "REFINEMENT_INCOMPLETE", "TOPOLOGY_UNRESOLVED", "DEADLINE_EXPIRED"]


type PlaneCurveArclengthOutcome = Annotated[
    ArclengthEnclosed | ArclengthEmpty | ArclengthSingularUnsupported | ArclengthUnknown,
    Field(discriminator="status"),
]


class PlaneCurveArclengthResult(PlaneCurveArclengthRequest):
    """Source-bound exact rational arclength result or typed non-conclusion."""

    outcome: PlaneCurveArclengthOutcome

    @model_validator(mode="after")
    def bind_result_source(self) -> Self:
        if isinstance(self.outcome, ArclengthEnclosed) and self.outcome.upper.as_fraction() - self.outcome.lower.as_fraction() > self.target_width.as_fraction():
            raise _validation_error("target_width", "ENCLOSED result exceeds the requested width")
        return self

    @classmethod
    def _from_kernel(cls, request: PlaneCurveArclengthRequest, *, outcome: PlaneCurveArclengthOutcome) -> Self:
        return cls.model_construct(
            polynomial=request.polynomial,
            box=request.box,
            target_width=request.target_width,
            resource_budget=request.resource_budget,
            outcome=outcome,
        )


__all__ = [
    "MAX_ARCLENGTH_BOX_ENDPOINT_DIGITS",
    "MAX_ARCLENGTH_COEFFICIENT_DIGITS",
    "MAX_ARCLENGTH_DEGREE",
    "MAX_ARCLENGTH_PRECISION_BITS",
    "MAX_ARCLENGTH_SEGMENTS",
    "MAX_ARCLENGTH_SOURCE_TERMS",
    "MAX_ARCLENGTH_WALL_SECONDS",
    "ArclengthEmpty",
    "ArclengthEnclosed",
    "ArclengthSegment",
    "ArclengthSingularUnsupported",
    "ArclengthUnknown",
    "PlaneCurveArclengthBudget",
    "PlaneCurveArclengthRequest",
    "PlaneCurveArclengthResult",
]
