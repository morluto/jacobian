"""Pinned distance to pair-spanned lines operations."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Literal

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
    def require_unique_points(self):
        keys = tuple(
            (p.x.num, p.x.den, p.y.num, p.y.den)
            for p in self.points
        )
        if len(keys) != len(set(keys)):
            raise ValueError("point-set coordinates must be unique")
        return self


class LineDistanceEntry(StrictModel):
    """One distinct line with its exact squared distance and source pairs."""

    squared_distance_numerator: str
    squared_distance_denominator: str
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
    def require_invariants(self):
        from fractions import Fraction
        from math import gcd

        from jacobian.canonical import format_canonical_integer

        if self.distinct_line_count != len(self.lines):
            raise ValueError("distinct_line_count must match the line count")
        if self.lines:
            min_entry = min(self.lines, key=lambda e: Fraction(
                parse_canonical_integer(e.squared_distance_numerator),
                parse_canonical_integer(e.squared_distance_denominator),
            ))
            if self.min_squared_distance is None:
                raise ValueError("min_squared_distance is required when lines exist")
            if (
                self.min_squared_distance.squared_distance_numerator != min_entry.squared_distance_numerator
                or self.min_squared_distance.squared_distance_denominator != min_entry.squared_distance_denominator
            ):
                raise ValueError("min_squared_distance must be the minimum entry")
        # Replay the bounded line ledger from the retained source values to ensure
        # the result is not forged: recompute the expected lines and compare.
        # This is the defining invariant for the pinned-distance profile.
        ax = self.anchor.x.as_fraction()
        ay = self.anchor.y.as_fraction()
        pts = [(p.x.as_fraction(), p.y.as_fraction()) for p in self.points]

        def _canonical_line_key(xi, yi, xj, yj):
            from fractions import Fraction
            from math import gcd

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

        expected_map: dict[tuple[str, str, str], tuple[str, str]] = {}
        # Also track source pairs for validation
        pair_map: dict[tuple[str, str, str], set[tuple[int, int]]] = {}
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
                sq_dist = Fraction(cross * cross, norm_sq)
                key = _canonical_line_key(xi, yi, xj, yj)
                num_str = format_canonical_integer(sq_dist.numerator)
                den_str = format_canonical_integer(sq_dist.denominator)
                if key not in expected_map:
                    expected_map[key] = (num_str, den_str)
                    pair_map[key] = set()
                else:
                    # The distance for a given line must be the same for all pairs that generate it
                    if expected_map[key] != (num_str, den_str):
                        raise ValueError("inconsistent distance for the same line")
                pair_map[key].add((i, j))

        if len(expected_map) != len(self.lines):
            raise ValueError("line count does not match the recomputed ledger")

        # Build a map from line key to entry for the result
        result_map: dict[tuple[str, str, str], LineDistanceEntry] = {}
        # We need to reconstruct the key for each result entry; but the result does not store the key,
        # only the distance and source pairs.  Instead, we can check that every result entry's distance
        # and pairs correspond to a recomputed line.
        # For each result entry, find a matching expected line by distance and pairs
        # This is a bit indirect, but we can check that the set of distances and pair sets matches.

        # Collect expected distances and pair sets
        expected_entries = set()
        for key, (num, den) in expected_map.items():
            pairs = tuple(sorted(pair_map[key]))
            expected_entries.add((num, den, pairs))

        result_entries = set()
        for entry in self.lines:
            pairs = tuple(sorted(entry.source_pairs))
            result_entries.add((entry.squared_distance_numerator, entry.squared_distance_denominator, pairs))

        if result_entries != expected_entries:
            raise ValueError("pinned-distance entries do not match the recomputed ledger from the source points and anchor")
        return self


def compute_pinned_distances(request: PinnedDistanceRequest) -> PinnedDistanceResult:
    """Compute exact squared distances from an anchor to all pair-spanned lines."""
    ax = request.anchor.x.as_fraction()
    ay = request.anchor.y.as_fraction()

    pts = [(p.x.as_fraction(), p.y.as_fraction()) for p in request.points]

    # For each pair (i, j), compute the line and the squared distance from anchor
    # Line through points i, j: direction (dx, dy) = (p_j - p_i)
    # Squared distance from point P to line through A with direction D:
    # d^2 = cross(D, P-A)^2 / |D|^2
    # Entries are keyed by the canonical integer line equation a*x + b*y = c so
    # that distinct lines are never merged; the distance is entry data.
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

    line_map: dict[
        tuple[str, str, str], tuple[list[tuple[int, int]], str, str]
    ] = {}

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
                line_map[key] = (
                    [],
                    format_canonical_integer(sq_dist.numerator),
                    format_canonical_integer(sq_dist.denominator),
                )
            line_map[key][0].append((i, j))

    entries = []
    for key in sorted(line_map.keys()):
        pairs, num, den = line_map[key]
        entry = LineDistanceEntry(
            squared_distance_numerator=num,
            squared_distance_denominator=den,
            source_pairs=tuple(pairs),
        )
        entries.append(entry)

    min_entry = min(entries, key=lambda e: Fraction(
        parse_canonical_integer(e.squared_distance_numerator),
        parse_canonical_integer(e.squared_distance_denominator),
    )) if entries else None

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
