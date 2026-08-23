"""Pinned distance to pair-spanned lines operations."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.catalog._examples import example
from jacobian.math.geometry._models import RationalPoint2D
from jacobian.math.geometry._support import geometry_operation

# With per-component digit bound d, each squared distance carries at most
# 8d+6 numerator and denominator digits and each canonical line key at most
# 4d+6 digits. The point bound caps pair enumeration at C(32,2) = 496
# pair-spanned lines (the result validation replays it), so one request stays
# well inside a practical output budget.
MAX_PINNED_DISTANCE_POINTS = 32
_MAX_PINNED_COORDINATE_DIGITS = 256


class PinnedDistanceRequest(StrictModel):
    """Compute the complete pinned-distance profile from an anchor to all pair-spanned lines."""

    anchor: RationalPoint2D
    points: tuple[RationalPoint2D, ...] = Field(
        min_length=2, max_length=MAX_PINNED_DISTANCE_POINTS
    )

    @model_validator(mode="after")
    def require_unique_points(self) -> Self:
        keys = tuple((p.x.num, p.x.den, p.y.num, p.y.den) for p in self.points)
        if len(keys) != len(set(keys)):
            raise ValueError("point-set coordinates must be unique")
        for pt in (self.anchor, *self.points):
            require_bounded_rational(
                pt.x, max_digits=_MAX_PINNED_COORDINATE_DIGITS, label="coordinate"
            )
            require_bounded_rational(
                pt.y, max_digits=_MAX_PINNED_COORDINATE_DIGITS, label="coordinate"
            )
        return self


class LineDistanceEntry(StrictModel):
    """One distinct line with its exact squared distance and source pairs."""

    squared_distance: CanonicalRational
    source_pairs: tuple[tuple[int, int], ...]


class PinnedDistanceResult(StrictModel):
    """The complete pinned-distance profile."""

    anchor: RationalPoint2D
    points: tuple[RationalPoint2D, ...]
    lines: tuple[LineDistanceEntry, ...]
    distinct_line_count: int = Field(ge=0)
    min_squared_distance: LineDistanceEntry | None = None
    complete: Literal[True] = True
    method: Literal["EXACT_PINNED_DISTANCES"] = "EXACT_PINNED_DISTANCES"

    @model_validator(mode="after")
    def require_invariants(self) -> Self:
        from jacobian._exact import require_bounded_rational

        # A directly supplied result bypasses PinnedDistanceRequest, so
        # reapply the full request cardinality, uniqueness, and
        # coordinate-height bounds before any replay work.
        if not 2 <= len(self.points) <= MAX_PINNED_DISTANCE_POINTS:
            raise ValueError(
                "point set must contain between 2 and "
                f"{MAX_PINNED_DISTANCE_POINTS} points"
            )
        keys = tuple((p.x.num, p.x.den, p.y.num, p.y.den) for p in self.points)
        if len(keys) != len(set(keys)):
            raise ValueError("point-set coordinates must be unique")
        for pt in (self.anchor, *self.points):
            require_bounded_rational(
                pt.x, max_digits=_MAX_PINNED_COORDINATE_DIGITS, label="coordinate"
            )
            require_bounded_rational(
                pt.y, max_digits=_MAX_PINNED_COORDINATE_DIGITS, label="coordinate"
            )
        if self.distinct_line_count != len(self.lines):
            raise ValueError("distinct_line_count must match the line count")
        # Replay the bounded computation from the retained points so an
        # authored profile cannot claim false distances, pairs, or counts.
        expected = _line_entries_from_points(self.anchor, self.points)
        if self.lines != expected:
            raise ValueError(
                "lines must equal the exact replay from the retained anchor and points"
            )
        if self.min_squared_distance is not None:
            if not self.lines:
                raise ValueError("no minimum entry can exist without lines")
            min_entry = min(
                expected,
                key=lambda e: e.squared_distance.as_fraction(),
            )
            if self.min_squared_distance != min_entry:
                raise ValueError("min_squared_distance must be the minimum entry")
        elif self.lines:
            raise ValueError("min_squared_distance is required when lines exist")
        return self


def _canonical_line_key(
    xi: Fraction, yi: Fraction, xj: Fraction, yj: Fraction
) -> tuple[str, str, str]:
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


def _line_entries_from_points(
    anchor: RationalPoint2D,
    points: tuple[RationalPoint2D, ...],
) -> tuple[LineDistanceEntry, ...]:
    """Replay the exact pair-spanned-line profile from retained geometry."""

    ax = anchor.x.as_fraction()
    ay = anchor.y.as_fraction()
    pts = [(p.x.as_fraction(), p.y.as_fraction()) for p in points]

    # For each pair (i, j), compute the line and the squared distance from anchor
    # Line through points i, j: direction (dx, dy) = (p_j - p_i)
    # Squared distance from point P to line through A with direction D:
    # d^2 = cross(D, P-A)^2 / |D|^2
    # Entries are keyed by the canonical integer line equation a*x + b*y = c so
    # that distinct lines are never merged; the distance is entry data.
    line_map: dict[tuple[str, str, str], tuple[list[tuple[int, int]], Fraction]] = {}

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            # Cross product of direction with anchor displacement
            cross = dx * (ay - yi) - dy * (ax - xi)
            norm_sq = dx * dx + dy * dy
            if norm_sq == 0:
                continue  # duplicate points shouldn't happen, but be safe
            sq_dist = Fraction(cross * cross, norm_sq)
            key = _canonical_line_key(xi, yi, xj, yj)
            if key not in line_map:
                line_map[key] = ([], sq_dist)
            line_map[key][0].append((i, j))

    entries = []
    for key in sorted(line_map.keys()):
        pairs, sq_dist = line_map[key]
        entries.append(
            LineDistanceEntry(
                squared_distance=CanonicalRational.from_fraction(sq_dist),
                source_pairs=tuple(pairs),
            )
        )
    return tuple(entries)


def compute_pinned_distances(request: PinnedDistanceRequest) -> PinnedDistanceResult:
    """Compute exact squared distances from an anchor to all pair-spanned lines."""

    entries = _line_entries_from_points(request.anchor, request.points)

    min_entry = (
        min(
            entries,
            key=lambda e: e.squared_distance.as_fraction(),
        )
        if entries
        else None
    )

    return PinnedDistanceResult(
        anchor=request.anchor,
        points=request.points,
        lines=tuple(entries),
        distinct_line_count=len(entries),
        min_squared_distance=min_entry,
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
