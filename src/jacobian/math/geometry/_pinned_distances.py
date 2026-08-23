"""Pinned distance to pair-spanned lines operations."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Literal

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.math.geometry._models import RationalPoint2D
from jacobian.math.geometry._support import geometry_operation

# Coordinate heights are bounded so every printed integer stays far below
# CPython's 4300-digit int->str conversion limit: canonical line keys scale
# by the LCM of up to three denominators (<= 4H+2 digits), and the reduced
# squared distance cross^2/norm^2 reaches roughly 12H+10 digits. At H = 256
# the largest component is about 3082 digits.
MAX_PINNED_COORDINATE_DIGITS = 256


def _require_bounded_unique_points(
    anchor: RationalPoint2D,
    points: tuple[RationalPoint2D, ...],
) -> None:
    """The pinned-distance admission bound shared by request and result.

    Canonical line keys are integer triples obtained by scaling with the
    LCM of up to three coordinate denominators; see the module constant
    above for the height derivation. Capping points at 32 keeps the
    aggregate quadratic ledger inside the canonical output ceiling: at most
    C(32,2)=496 entries of ~3000-digit distances.
    """
    from jacobian.math._rational_height import RationalHeight

    for value in (anchor.x, anchor.y):
        if RationalHeight.from_canonical(value).exceeds(MAX_PINNED_COORDINATE_DIGITS):
            raise ValueError(
                "anchor coordinates exceed the conservative "
                f"{MAX_PINNED_COORDINATE_DIGITS}-digit pinned-distance bound"
            )
    keys = tuple((p.x.num, p.x.den, p.y.num, p.y.den) for p in points)
    if len(keys) != len(set(keys)):
        raise ValueError("point-set coordinates must be unique")
    for point in points:
        for value in (point.x, point.y):
            if RationalHeight.from_canonical(value).exceeds(
                MAX_PINNED_COORDINATE_DIGITS
            ):
                raise ValueError(
                    "point coordinates exceed the conservative "
                    f"{MAX_PINNED_COORDINATE_DIGITS}-digit pinned-distance bound"
                )


class PinnedDistanceRequest(StrictModel):
    """Compute the complete pinned-distance profile from an anchor to all pair-spanned lines."""

    anchor: RationalPoint2D
    points: tuple[RationalPoint2D, ...] = Field(min_length=2, max_length=32)

    @model_validator(mode="after")
    def require_unique_bounded_points(self):
        _require_bounded_unique_points(self.anchor, self.points)
        return self


class LineDistanceEntry(StrictModel):
    """One distinct line with its exact squared distance and source pairs."""

    squared_distance: CanonicalRational
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
    from jacobian.canonical import format_canonical_integer

    return (
        format_canonical_integer(int_a),
        format_canonical_integer(int_b),
        format_canonical_integer(int_c),
    )


def _distance_ledger(
    anchor: RationalPoint2D,
    points: tuple[RationalPoint2D, ...],
) -> tuple[LineDistanceEntry, ...]:
    """Exact pinned-distance profile shared by execution and validation.

    For each pair (i, j), the line through points i and j carries squared
    anchor distance ``cross(D, P-A)^2 / |D|^2``; entries are keyed by the
    canonical integer line equation so distinct lines are never merged.
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


class PinnedDistanceResult(StrictModel):
    """The complete pinned-distance profile."""

    anchor: RationalPoint2D
    points: tuple[RationalPoint2D, ...] = Field(min_length=2, max_length=32)
    lines: tuple[LineDistanceEntry, ...]
    distinct_line_count: int = Field(ge=0)
    min_squared_distance: LineDistanceEntry | None = None
    complete: Literal[True] = True
    method: Literal["EXACT_PINNED_DISTANCES"] = "EXACT_PINNED_DISTANCES"

    @model_validator(mode="after")
    def require_invariants(self):
        if self.distinct_line_count != len(self.lines):
            raise ValueError("distinct_line_count must match the line count")
        # Retained sources satisfy the same admission bound as a request, so
        # an independently decoded profile can neither carry impossible point
        # sets nor replay the quadratic ledger outside the operation's bound.
        _require_bounded_unique_points(self.anchor, self.points)
        # Source-bound replay: recompute the canonical line ledger from the
        # retained anchor and points and compare every entry, so a relayed
        # or truncated profile (e.g. missing lines for retained points)
        # cannot validate.
        expected = _distance_ledger(self.anchor, self.points)
        if tuple(self.lines) != expected:
            raise ValueError(
                "pinned-distance lines must be the exact canonical ledger "
                "of the retained anchor and points"
            )
        if not expected:
            if self.min_squared_distance is not None:
                raise ValueError("an empty line ledger carries no minimum entry")
            return self
        min_entry = min(
            expected,
            key=lambda e: e.squared_distance.as_fraction(),
        )
        if self.min_squared_distance is None:
            raise ValueError("min_squared_distance is required when lines exist")
        if self.min_squared_distance != min_entry:
            raise ValueError("min_squared_distance must be the minimum entry")
        return self


def compute_pinned_distances(request: PinnedDistanceRequest) -> PinnedDistanceResult:
    """Compute exact squared distances from an anchor to all pair-spanned lines."""
    entries = _distance_ledger(request.anchor, request.points)

    min_entry = (
        min(entries, key=lambda e: e.squared_distance.as_fraction())
        if entries
        else None
    )

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
