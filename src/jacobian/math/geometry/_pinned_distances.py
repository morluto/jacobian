"""Pinned distance to pair-spanned lines operations."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import require_bounded_rational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog._examples import example
from jacobian.math.geometry._models import RationalPoint2D
from jacobian.math.geometry._support import geometry_operation

# With per-component digit bound d, each squared-distance fraction carries at
# most 8d+6 numerator and denominator digits, so d = 2048 keeps every exact
# entry inside the canonical 32,768-digit limit before execution and safely
# below any integer-string conversion boundary in formatting.
_MAX_PINNED_COORDINATE_DIGITS = 2048


class PinnedDistanceRequest(StrictModel):
    """Compute the complete pinned-distance profile from an anchor to all pair-spanned lines."""

    anchor: RationalPoint2D
    points: tuple[RationalPoint2D, ...] = Field(min_length=2, max_length=128)

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


def _canonical_line_key(
    xi: Fraction, yi: Fraction, xj: Fraction, yj: Fraction
) -> tuple[str, str, str]:
    """Canonical integer line equation a*x + b*y = c for the pair-spanned line."""
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


def _distance_ledger(
    anchor: RationalPoint2D, points: tuple[RationalPoint2D, ...]
) -> list[LineDistanceEntry]:
    """Build the exact sorted pinned-distance ledger from the configuration.

    Shared by execution and result validation so both paths construct the
    identical canonical entries: distinct lines keyed by the canonical
    integer line equation, pairs grouped per line, distances exact.
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
            cross = dx * (ay - yi) - dy * (ax - xi)
            norm_sq = dx * dx + dy * dy
            if norm_sq == 0:
                continue
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


def _entry_distance(entry: LineDistanceEntry) -> Fraction:
    """Exact squared distance of a ledger entry from its canonical strings."""
    return Fraction(
        parse_canonical_integer(entry.squared_distance_numerator),
        parse_canonical_integer(entry.squared_distance_denominator),
    )


def _require_matching_entry(
    expected: LineDistanceEntry, actual: LineDistanceEntry
) -> None:
    if (
        expected.squared_distance_numerator != actual.squared_distance_numerator
        or expected.squared_distance_denominator != actual.squared_distance_denominator
    ):
        raise ValueError("squared distance must match the exact pinned distance")
    if set(expected.source_pairs) != set(actual.source_pairs):
        raise ValueError("source_pairs must cover the exact line pairs")
    if tuple(sorted(actual.source_pairs)) != actual.source_pairs:
        raise ValueError("source_pairs must be sorted")


def _require_honest_minimum(
    lines: tuple[LineDistanceEntry, ...],
    minimum: LineDistanceEntry | None,
) -> None:
    if not lines:
        if minimum is not None:
            raise ValueError("empty line set cannot carry a minimum distance")
        # Every admissible request has at least two distinct points, hence at
        # least one spanned line; an empty line set is never valid.
        raise ValueError("pinned-distance profile must contain at least one line")
    min_entry = min(lines, key=_entry_distance)
    if minimum is None:
        raise ValueError("min_squared_distance is required when lines exist")
    # The reported minimum is a full LineDistanceEntry: it must equal an
    # actual minimum line entry, including its source pairs, so a detached
    # payload cannot rebind the ledger's identity.
    if minimum != min_entry:
        raise ValueError("min_squared_distance must be an actual minimum line entry")


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
    def require_invariants(self) -> Self:
        if self.distinct_line_count != len(self.lines):
            raise ValueError("distinct_line_count must match the line count")
        expected_entries = _distance_ledger(self.anchor, self.points)
        if len(expected_entries) != len(self.lines):
            raise ValueError("line count must match the exact pair-spanned lines")
        # Exact line entries must match in sorted-key order.
        for expected, actual in zip(expected_entries, self.lines, strict=True):
            _require_matching_entry(expected, actual)
        _require_honest_minimum(self.lines, self.min_squared_distance)
        return self


def compute_pinned_distances(request: PinnedDistanceRequest) -> PinnedDistanceResult:
    """Compute exact squared distances from an anchor to all pair-spanned lines.

    Entries are keyed by the canonical integer line equation a*x + b*y = c so
    that distinct lines are never merged; the distance is entry data. The
    shared ledger helper keeps execution and validation on one construction.
    """
    entries = _distance_ledger(request.anchor, request.points)
    min_entry = min(entries, key=_entry_distance) if entries else None

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
