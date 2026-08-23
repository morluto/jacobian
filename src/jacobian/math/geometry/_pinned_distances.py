"""Pinned distance to pair-spanned lines operations."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog._examples import example
from jacobian.math.geometry._models import RationalPoint2D
from jacobian.math.geometry._support import geometry_operation


class PinnedDistanceRequest(StrictModel):
    """Compute the complete pinned-distance profile from an anchor to all pair-spanned lines."""

    anchor: RationalPoint2D
    points: tuple[RationalPoint2D, ...] = Field(min_length=2, max_length=128)

    @model_validator(mode="after")
    def require_unique_points(self) -> Self:
        keys = tuple((p.x.num, p.x.den, p.y.num, p.y.den) for p in self.points)
        if len(keys) != len(set(keys)):
            raise ValueError("point-set coordinates must be unique")
        return self


class LineDistanceEntry(StrictModel):
    """One distinct line with its exact squared distance and source pairs."""

    squared_distance_numerator: str
    squared_distance_denominator: str
    source_pairs: tuple[tuple[int, int], ...]


def _canonical_line_key(
    xi: Fraction, yi: Fraction, xj: Fraction, yj: Fraction
) -> tuple[str, str, str]:
    """Canonical integer line equation a*x + b*y = c through two points."""
    dx = xj - xi
    dy = yj - yi
    # Normal form: dy*(X - xi) - dx*(Y - yi) = 0  =>  a*X + b*Y = c
    a = dy
    b = -dx
    c = a * xi + b * yi
    scale_lcm = 1
    for component in (a.denominator, b.denominator, c.denominator):
        scale_lcm = scale_lcm * component // gcd(scale_lcm, component)
    int_a = int(a * scale_lcm)
    int_b = int(b * scale_lcm)
    int_c = int(c * scale_lcm)
    divisor = gcd(gcd(abs(int_a), abs(int_b)), abs(int_c))
    if divisor:
        int_a //= divisor
        int_b //= divisor
        int_c //= divisor
    if int_a < 0 or (int_a == 0 and int_b < 0):
        int_a, int_b, int_c = -int_a, -int_b, -int_c
    return (
        format_canonical_integer(int_a),
        format_canonical_integer(int_b),
        format_canonical_integer(int_c),
    )


def _entry_distance(entry: LineDistanceEntry) -> Fraction:
    return Fraction(
        parse_canonical_integer(entry.squared_distance_numerator),
        parse_canonical_integer(entry.squared_distance_denominator),
    )


def _pinned_distance_entries(
    anchor: RationalPoint2D,
    points: tuple[RationalPoint2D, ...],
) -> tuple[LineDistanceEntry, ...]:
    """Recompute the canonical line ledger from one anchor and point set."""
    ax = anchor.x.as_fraction()
    ay = anchor.y.as_fraction()
    pts = [(p.x.as_fraction(), p.y.as_fraction()) for p in points]

    # Squared distance from point P to line through A with direction D:
    # d^2 = cross(D, P-A)^2 / |D|^2.  Entries are keyed by the canonical
    # integer line equation so distinct lines are never merged; the distance
    # and every source pair that generates the line are entry data.
    line_map: dict[tuple[str, str, str], tuple[list[tuple[int, int]], str, str]] = {}
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            cross = dx * (ay - yi) - dy * (ax - xi)
            norm_sq = dx * dx + dy * dy
            sq_dist = Fraction(cross * cross, norm_sq)
            key = _canonical_line_key(xi, yi, xj, yj)
            if key not in line_map:
                line_map[key] = (
                    [],
                    format_canonical_integer(sq_dist.numerator),
                    format_canonical_integer(sq_dist.denominator),
                )
            line_map[key][0].append((i, j))

    return tuple(
        LineDistanceEntry(
            squared_distance_numerator=num,
            squared_distance_denominator=den,
            source_pairs=tuple(pairs),
        )
        for key in sorted(line_map.keys())
        for pairs, num, den in (line_map[key],)
    )


class PinnedDistanceResult(StrictModel):
    """The complete pinned-distance profile bound to its retained sources."""

    anchor: RationalPoint2D
    points: tuple[RationalPoint2D, ...]
    lines: tuple[LineDistanceEntry, ...]
    distinct_line_count: int = Field(ge=0)
    min_squared_distance: LineDistanceEntry | None = None
    complete: Literal[True] = True
    method: Literal["EXACT_PINNED_DISTANCES"] = "EXACT_PINNED_DISTANCES"

    @model_validator(mode="after")
    def require_source_bound_profile(self) -> Self:
        keys = tuple((p.x.num, p.x.den, p.y.num, p.y.den) for p in self.points)
        if len(keys) != len(set(keys)):
            raise ValueError("point-set coordinates must be unique")
        expected = _pinned_distance_entries(self.anchor, self.points)
        if self.lines != expected:
            raise ValueError(
                "lines must equal the exact profile recomputed from the "
                "anchor and points"
            )
        if self.distinct_line_count != len(expected):
            raise ValueError("distinct_line_count must match the line count")
        min_entry = min(expected, key=_entry_distance) if expected else None
        if self.min_squared_distance != min_entry:
            raise ValueError("min_squared_distance must be the minimum entry")
        return self


def compute_pinned_distances(request: PinnedDistanceRequest) -> PinnedDistanceResult:
    """Compute exact squared distances from an anchor to all pair-spanned lines."""
    entries = _pinned_distance_entries(request.anchor, request.points)
    return PinnedDistanceResult(
        anchor=request.anchor,
        points=request.points,
        lines=entries,
        distinct_line_count=len(entries),
        min_squared_distance=(min(entries, key=_entry_distance) if entries else None),
    )


PINNED_DISTANCE_OPERATIONS = (
    geometry_operation(
        "geometry.points.compute.pinned_distances",
        "Compute pinned distances to pair-spanned lines",
        "Given a bounded labelled rational point configuration and a rational "
        "anchor, return the complete exact squared-distance profile from the "
        "anchor to every distinct line spanned by point pairs, retaining "
        "every source pair that generates each line.",
        PinnedDistanceRequest,
        PinnedDistanceResult,
        compute_pinned_distances,
        "geometry",
        "distance",
        examples=(
            example(
                "unit_square_anchors",
                "Compute pinned distances from an anchor to lines of a unit square.",
                {
                    "anchor": {
                        "x": {"num": "0", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                    "points": [
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
                        {"x": {"num": "1", "den": "1"}, "y": {"num": "0", "den": "1"}},
                        {"x": {"num": "1", "den": "1"}, "y": {"num": "1", "den": "1"}},
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "1", "den": "1"}},
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["PINNED_DISTANCE_OPERATIONS"]
