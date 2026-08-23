"""Pinned distance to pair-spanned lines operations."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from math import gcd
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog._examples import example
from jacobian.math._rational_height import RationalHeight
from jacobian.math.geometry._models import RationalPoint2D
from jacobian.math.geometry._support import geometry_operation

# Conservative worst-case propagation for the complete pinned-distance
# profile transport. Coordinates are rationals whose numerator and denominator
# have at most H digits; each coordinate difference reaches 2H+1 digits over
# 2H, the anchor cross product reaches ~4H+4 digits, its square ~8H+8, and the
# reduced squared distance therefore stays within ~12H+13 digits per
# component. The bound must cover the AGGREGATE profile: C(128,2) = 8128 line
# entries of two components plus per-entry JSON overhead (~100 bytes), the
# retained configuration and the repeated minimum entry must fit inside the
# canonical 10 MiB output limit:
# 8128 * (2*(12*32+13) + 100) ~= 7.3 MB < 10 MiB, so H = 32.
MAX_PINNED_COORDINATE_DIGITS = 32


def _pinned_coordinate_height_ok(point: RationalPoint2D) -> bool:
    for v in (point.x, point.y):
        if RationalHeight.from_canonical(v).exceeds(MAX_PINNED_COORDINATE_DIGITS):
            return False
    return True


def _canonical_line_key(
    xi: Fraction, yi: Fraction, xj: Fraction, yj: Fraction
) -> tuple[str, str, str]:
    """Canonical integer coefficients (a, b, c) of the line a*X + b*Y = c."""
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


def _exact_distance_entries(
    anchor: RationalPoint2D,
    points: tuple[RationalPoint2D, ...],
) -> list[LineDistanceEntry]:
    """Replay the complete exact pinned-distance ledger from retained sources.

    Distinct lines are keyed by their canonical integer equation so collinear
    pairs merge without merging distinct lines; each entry carries the exact
    squared distance and every ordered source pair generating the line.
    """
    ax = anchor.x.as_fraction()
    ay = anchor.y.as_fraction()
    pts = [(p.x.as_fraction(), p.y.as_fraction()) for p in points]
    line_map: dict[tuple[str, str, str], tuple[list[tuple[int, int]], str, str]] = {}
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            # d^2 = cross(D, P-A)^2 / |D|^2 for direction D and displacement P-A
            cross = dx * (ay - yi) - dy * (ax - xi)
            norm_sq = dx * dx + dy * dy
            if norm_sq == 0:
                raise ValueError("spanned lines require pairwise distinct points")
            sq_dist = Fraction(cross * cross, norm_sq)
            key = _canonical_line_key(xi, yi, xj, yj)
            if key not in line_map:
                line_map[key] = (
                    [],
                    format_canonical_integer(sq_dist.numerator),
                    format_canonical_integer(sq_dist.denominator),
                )
            line_map[key][0].append((i, j))
    return [
        LineDistanceEntry(
            squared_distance_numerator=num,
            squared_distance_denominator=den,
            source_pairs=tuple(pairs),
        )
        for key in sorted(line_map.keys())
        for pairs, num, den in (line_map[key],)
    ]


def _minimum_entry(
    entries: Sequence[LineDistanceEntry],
) -> LineDistanceEntry | None:
    if not entries:
        return None
    return min(
        entries,
        key=lambda e: Fraction(
            parse_canonical_integer(e.squared_distance_numerator),
            parse_canonical_integer(e.squared_distance_denominator),
        ),
    )


def _require_pinned_source_admission(
    anchor: RationalPoint2D, points: tuple[RationalPoint2D, ...]
) -> None:
    """Apply the request's point-count-uniqueness and coordinate-height bounds."""
    keys = tuple((p.x.num, p.x.den, p.y.num, p.y.den) for p in points)
    if len(keys) != len(set(keys)):
        raise ValueError("point-set coordinates must be unique")
    for point in (anchor, *points):
        if not _pinned_coordinate_height_ok(point):
            raise ValueError(
                "pinned-distance coordinates exceed the conservative "
                f"{MAX_PINNED_COORDINATE_DIGITS}-digit input bound that "
                "keeps the complete profile inside transport"
            )


class PinnedDistanceRequest(StrictModel):
    """Compute the complete pinned-distance profile from an anchor to all pair-spanned lines."""

    anchor: RationalPoint2D
    points: tuple[RationalPoint2D, ...] = Field(min_length=2, max_length=128)

    @model_validator(mode="after")
    def require_unique_points(self) -> Self:
        _require_pinned_source_admission(self.anchor, self.points)
        return self


class LineDistanceEntry(StrictModel):
    """One distinct line with its exact squared distance and source pairs."""

    squared_distance_numerator: str
    squared_distance_denominator: str
    source_pairs: tuple[tuple[int, int], ...]


class PinnedDistanceResult(StrictModel):
    """The complete pinned-distance profile."""

    anchor: RationalPoint2D
    points: tuple[RationalPoint2D, ...] = Field(min_length=2, max_length=128)
    lines: tuple[LineDistanceEntry, ...]
    distinct_line_count: int = Field(ge=0)
    min_squared_distance: LineDistanceEntry | None = None
    complete: Literal[True] = True
    method: Literal["EXACT_PINNED_DISTANCES"] = "EXACT_PINNED_DISTANCES"

    @model_validator(mode="after")
    def require_invariants(self) -> Self:
        # Retained sources revalidate through the request admission before the
        # quadratic pair enumeration replays, so a serialized profile can never
        # carry more points or taller coordinates than a fresh request.
        _require_pinned_source_admission(self.anchor, self.points)
        if self.distinct_line_count != len(self.lines):
            raise ValueError("distinct_line_count must match the line count")
        expected_entries = _exact_distance_entries(self.anchor, self.points)
        if len(expected_entries) != len(self.lines):
            raise ValueError("line count must match the exact pair-spanned lines")
        for expected, actual in zip(expected_entries, self.lines, strict=True):
            if (
                expected.squared_distance_numerator != actual.squared_distance_numerator
                or expected.squared_distance_denominator
                != actual.squared_distance_denominator
            ):
                raise ValueError(
                    "squared distance must match the exact pinned distance"
                )
            if set(expected.source_pairs) != set(actual.source_pairs):
                raise ValueError("source_pairs must cover the exact line pairs")
            if tuple(sorted(actual.source_pairs)) != actual.source_pairs:
                raise ValueError("source_pairs must be sorted")
        minimum = _minimum_entry(self.lines)
        if minimum is None:
            raise ValueError("pinned-distance profile must contain at least one line")
        if self.min_squared_distance is None:
            raise ValueError("min_squared_distance is required when lines exist")
        # The reported minimum must be the complete canonical entry selected
        # from the replayed lines, including the source pairs that generate it.
        if self.min_squared_distance != minimum:
            raise ValueError("min_squared_distance must be the selected minimum entry")
        return self


def compute_pinned_distances(request: PinnedDistanceRequest) -> PinnedDistanceResult:
    """Compute exact squared distances from an anchor to all pair-spanned lines."""
    entries = _exact_distance_entries(request.anchor, request.points)
    return PinnedDistanceResult(
        anchor=request.anchor,
        points=request.points,
        lines=tuple(entries),
        distinct_line_count=len(entries),
        min_squared_distance=_minimum_entry(entries),
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
