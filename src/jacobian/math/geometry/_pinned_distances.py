"""Pinned distance to pair-spanned lines operations."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import gcd
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.catalog._examples import example
from jacobian.math._rational_height import RationalHeight, sum_heights
from jacobian.math.geometry._models import RationalPoint2D
from jacobian.math.geometry._support import geometry_operation


class PinnedDistanceRequest(StrictModel):
    """Compute the complete pinned-distance profile from an anchor to all pair-spanned lines."""

    anchor: RationalPoint2D
    points: tuple[RationalPoint2D, ...] = Field(min_length=2, max_length=128)

    @model_validator(mode="after")
    def require_admissible_points(self) -> Self:
        keys = tuple((p.x.num, p.x.den, p.y.num, p.y.den) for p in self.points)
        if len(keys) != len(set(keys)):
            raise ValueError("point-set coordinates must be unique")
        # d^2 = cross(D, P-A)^2 / |D|^2 propagates coordinate heights into a
        # squared-distance height far larger than any flat multiple of the
        # coordinate cap once differences square and divide, so propagate a
        # conservative rational-height bound for every spanned pair instead.
        ax = RationalHeight.from_canonical(self.anchor.x)
        ay = RationalHeight.from_canonical(self.anchor.y)
        xs = [RationalHeight.from_canonical(p.x) for p in self.points]
        ys = [RationalHeight.from_canonical(p.y) for p in self.points]
        for i, j in combinations(range(len(self.points)), 2):
            dx = sum_heights((xs[i], xs[j]))
            dy = sum_heights((ys[i], ys[j]))
            cross = sum_heights(
                (
                    dx.product(sum_heights((ys[i], ay))),
                    dy.product(sum_heights((xs[i], ax))),
                )
            )
            norm_sq = sum_heights((dx.product(dx), dy.product(dy)))
            if (
                cross.product(cross)
                .quotient(norm_sq)
                .exceeds(MAX_CANONICAL_RATIONAL_DIGITS)
            ):
                raise ValueError(
                    f"pair ({i}, {j}) has a squared-distance height exceeding "
                    f"the canonical {MAX_CANONICAL_RATIONAL_DIGITS}-digit limit"
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
        expected_entries = _line_ledger(self.anchor, self.points)
        if self.distinct_line_count != len(expected_entries):
            raise ValueError("distinct_line_count must match the line count")
        if self.lines != expected_entries:
            raise ValueError(
                "pinned-distance entries do not match the recomputed ledger "
                "from the source points and anchor"
            )
        if expected_entries:
            min_entry = min(expected_entries, key=_entry_distance)
            if self.min_squared_distance != min_entry:
                raise ValueError(
                    "min_squared_distance must be a minimum ledger entry with its source pairs"
                )
        return self


def _entry_distance(entry: LineDistanceEntry) -> Fraction:
    return entry.squared_distance.as_fraction()


def _canonical_line_key(
    xi: Fraction, yi: Fraction, xj: Fraction, yj: Fraction
) -> tuple[str, str, str]:
    """Canonical integer line equation a*X + b*Y = c through two points."""
    dx = xj - xi
    dy = yj - yi
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


def _line_ledger(
    anchor: RationalPoint2D, points: tuple[RationalPoint2D, ...]
) -> tuple[LineDistanceEntry, ...]:
    """Replay the bounded line ledger from the retained source values."""
    ax = anchor.x.as_fraction()
    ay = anchor.y.as_fraction()
    pts = [(p.x.as_fraction(), p.y.as_fraction()) for p in points]

    line_map: dict[
        tuple[str, str, str], tuple[CanonicalRational, list[tuple[int, int]]]
    ] = {}
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            cross = dx * (ay - yi) - dy * (ax - xi)
            norm_sq = dx * dx + dy * dy
            if norm_sq == 0:
                continue
            sq_dist = CanonicalRational.from_fraction(Fraction(cross * cross, norm_sq))
            key = _canonical_line_key(xi, yi, xj, yj)
            if key not in line_map:
                line_map[key] = (sq_dist, [])
            elif line_map[key][0] != sq_dist:
                raise ValueError("inconsistent distance for the same line")
            line_map[key][1].append((i, j))
    return tuple(
        LineDistanceEntry(squared_distance=sq_dist, source_pairs=tuple(pairs))
        for _, (sq_dist, pairs) in sorted(line_map.items())
    )


def compute_pinned_distances(request: PinnedDistanceRequest) -> PinnedDistanceResult:
    """Compute exact squared distances from an anchor to all pair-spanned lines."""
    entries = _line_ledger(request.anchor, request.points)
    min_entry = min(entries, key=_entry_distance) if entries else None

    return PinnedDistanceResult(
        anchor=request.anchor,
        points=request.points,
        lines=entries,
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
