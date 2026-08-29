"""Domain functions for projective coordinate operations."""

from __future__ import annotations

from fractions import Fraction

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.projective.coordinates._models import (
    MAX_PROJECTIVE_COORDINATE_DIGITS,
    ChartTransitionResult,
    RationalPointConstructResult,
    RationalProjectivePoint,
    StandardChartResult,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"geometry.{reason}", message)


def _require_ratio_result_budget(
    coordinates: tuple[CanonicalRational, ...],
) -> None:
    if any(
        len(component.lstrip("-")) > MAX_PROJECTIVE_COORDINATE_DIGITS
        for coordinate in coordinates
        for component in (coordinate.num, coordinate.den)
    ):
        raise _validation_error(
            "projective_coordinate_components_exceed_digit_ratio",
            "projective coordinate components exceed the "
            f"{MAX_PROJECTIVE_COORDINATE_DIGITS:,}-digit ratio budget",
        )


def _admit_coordinates(coordinates: tuple[CanonicalRational, ...]) -> None:
    try:
        _require_ratio_result_budget(coordinates)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("request",), code=exc.type, message=exc.message()
        ) from exc


def _reject(reason: str, message: str, location: tuple[str, ...]) -> None:
    raise OperationDomainValidationError(
        location=location, code=f"geometry.{reason}", message=message
    )


def _rational(frac: Fraction) -> CanonicalRational:
    return CanonicalRational(
        num=format_canonical_integer(frac.numerator),
        den=format_canonical_integer(frac.denominator),
    )


def rational_projective_point(
    coordinates: tuple[CanonicalRational, ...],
) -> RationalPointConstructResult:
    """Canonicalize by scaling so first nonzero coordinate is 1."""
    _admit_coordinates(coordinates)
    if all(c.as_fraction() == 0 for c in coordinates):
        _reject(
            "projective_point_least_nonzero_coordinate",
            "projective point must have at least one nonzero coordinate",
            ("coordinates",),
        )
    coords = coordinates
    for _i, c in enumerate(coords):
        if c.as_fraction() != 0:
            inv = Fraction(1, 1) / c.as_fraction()
            scale = _rational(inv)
            canonical = tuple(_rational(v.as_fraction() * inv) for v in coords)
            return RationalPointConstructResult(
                point=RationalProjectivePoint(coordinates=canonical),
                scale=scale,
            )
    raise AssertionError("admission accepted all-zero projective coordinates")


def standard_chart(
    point: RationalProjectivePoint, chart_index: int
) -> StandardChartResult:
    """Dehomogenize at the given chart index (divide by that coordinate)."""
    coords = point.coordinates
    _admit_coordinates(coords)
    if chart_index >= len(coords):
        _reject("chart_index_out_range", "chart_index out of range", ("chart_index",))
    if coords[chart_index].as_fraction() == 0:
        _reject(
            "chart_coordinate_nonzero",
            "chart coordinate must be nonzero",
            ("chart_index",),
        )
    chart = chart_index
    inv = Fraction(1, 1) / coords[chart].as_fraction()
    affine = tuple(
        _rational(coords[i].as_fraction() * inv)
        for i in range(len(coords))
        if i != chart
    )
    return StandardChartResult(
        affine_point=affine,
        chart_index=chart,
    )


def chart_transition(
    point: RationalProjectivePoint, chart_i: int, chart_j: int
) -> ChartTransitionResult:
    """Return the complete target-chart coordinates for the projective point."""
    coords = point.coordinates
    _admit_coordinates(coords)
    if chart_i >= len(coords) or chart_j >= len(coords):
        _reject(
            "chart_index_out_range", "chart index out of range", ("chart_i", "chart_j")
        )
    if coords[chart_i].as_fraction() == 0:
        _reject(
            "chart_i_coordinate_nonzero",
            "chart_i coordinate must be nonzero",
            ("chart_i",),
        )
    xj = coords[chart_j].as_fraction()
    if xj == 0:
        return ChartTransitionResult(
            status="OUTSIDE_TARGET_CHART",
            transition=None,
            chart_i=chart_i,
            chart_j=chart_j,
            projective_dimension=len(coords) - 1,
        )
    ratios = tuple(
        _rational(coords[i].as_fraction() / xj)
        for i in range(len(coords))
        if i != chart_j
    )
    return ChartTransitionResult(
        status="DEFINED",
        transition=ratios,
        chart_i=chart_i,
        chart_j=chart_j,
        projective_dimension=len(coords) - 1,
    )


__all__ = ["chart_transition", "rational_projective_point", "standard_chart"]
