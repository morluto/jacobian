"""Domain functions for projective coordinate operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.projective_coords_ops._models import (
    ChartTransitionRequest,
    ChartTransitionResult,
    RationalPointConstructRequest,
    RationalPointConstructResult,
    StandardChartRequest,
    StandardChartResult,
)


def _rational(frac: Fraction) -> CanonicalRational:
    return CanonicalRational(
        num=format_canonical_integer(frac.numerator),
        den=format_canonical_integer(frac.denominator),
    )


def compute_rational_point_construct(
    request: RationalPointConstructRequest,
) -> RationalPointConstructResult:
    """Canonicalize by scaling so first nonzero coordinate is 1."""
    coords = request.coordinates
    for _i, c in enumerate(coords):
        if c.as_fraction() != 0:
            inv = Fraction(1, 1) / c.as_fraction()
            scale = _rational(inv)
            canonical = tuple(_rational(v.as_fraction() * inv) for v in coords)
            return RationalPointConstructResult(
                canonical=canonical,
                scale=scale,
                projective_dimension=len(coords) - 1,
            )
    raise ValueError("all coordinates are zero")


def compute_standard_chart(request: StandardChartRequest) -> StandardChartResult:
    """Dehomogenize at the given chart index (divide by that coordinate)."""
    coords = request.point.coordinates
    chart = request.chart_index
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


def compute_chart_transition(request: ChartTransitionRequest) -> ChartTransitionResult:
    """Compute the transition map from chart_i to chart_j coordinates."""
    coords = request.point.coordinates
    xi = coords[request.chart_i].as_fraction()
    xj = coords[request.chart_j].as_fraction()
    ratios = tuple(
        _rational(coords[i].as_fraction() * xi / xj)
        for i in range(len(coords))
        if i != request.chart_i and i != request.chart_j
    )
    return ChartTransitionResult(
        transition=ratios,
        chart_i=request.chart_i,
        chart_j=request.chart_j,
    )
