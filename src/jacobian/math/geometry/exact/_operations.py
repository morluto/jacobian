"""Domain-owned exact geometry operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.exact._models import (
    DistanceGraphRequest,
    DistanceGraphResult,
    DistanceMultiplicityEntry,
    DistanceProfileRequest,
    DistanceProfileResult,
    LabelledRationalPoint,
    PinnedLineDistanceRequest,
    PinnedLineDistanceResult,
)


def _to_fraction_point(point: LabelledRationalPoint) -> tuple[Fraction, ...]:
    return tuple(c.as_fraction() for c in point.coordinates)


def _squared_distance(
    p: tuple[Fraction, ...],
    q: tuple[Fraction, ...],
) -> Fraction:
    result = Fraction(0)
    for a, b in zip(p, q, strict=True):
        result += (a - b) ** 2
    return result


def compute_distance_profile(
    request: DistanceProfileRequest,
) -> DistanceProfileResult:
    """Compute exact pairwise squared distances for every unordered pair."""
    config = request.configuration
    n = len(config.points)
    dim = len(config.points[0].coordinates)
    points = [_to_fraction_point(p) for p in config.points]

    from collections import Counter

    distances: Counter[Fraction] = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            d = _squared_distance(points[i], points[j])
            distances[d] += 1

    entries = tuple(
        DistanceMultiplicityEntry(
            squared_distance=CanonicalRational.from_fraction(d),
            pair_count=count,
        )
        for d, count in sorted(distances.items())
    )
    return DistanceProfileResult(
        dimension=dim,
        point_count=n,
        entries=entries,
    )


def compute_distance_graph(
    request: DistanceGraphRequest,
) -> DistanceGraphResult:
    """Build the graph whose edges connect pairs at the target squared distance."""
    config = request.configuration
    n = len(config.points)
    points = [_to_fraction_point(p) for p in config.points]
    target = request.target_squared_distance.as_fraction()

    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if _squared_distance(points[i], points[j]) == target:
                edges.append((i, j))

    return DistanceGraphResult(
        vertex_count=n,
        edges=tuple(edges),
    )


__all__ = [
    "compute_distance_graph",
    "compute_distance_profile",
    "compute_pinned_line_distance_profile",
]


def _canonical_line_coefficients(
    p: tuple[Fraction, ...],
    q: tuple[Fraction, ...],
) -> tuple[Fraction, Fraction, Fraction]:
    """Return the sign- and gcd-normalized coefficients (A, B, C) of Ax+By+C=0.

    The line through two distinct points p, q has normal (dy, -dx) where
    (dx, dy) = q - p, so A = dy, B = -dx, C = -(A*px + B*py).  The triple is
    reduced by the gcd of A, B, C and its leading nonzero coefficient is made
    positive, giving a canonical identity for the geometric line.
    """
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    a = dy
    b = -dx
    c = -(a * p[0] + b * p[1])
    g = _gcd3(a, b, c)
    if g != 0:
        a, b, c = a / g, b / g, c / g
    # Normalize the sign so the first nonzero coefficient is positive.
    for coeff in (a, b, c):
        if coeff != 0:
            if coeff < 0:
                a, b, c = -a, -b, -c
            break
    return a, b, c


def _gcd3(a: Fraction, b: Fraction, c: Fraction) -> Fraction:
    """Return a positive Fraction that divides a, b, c (the gcd of numerators)."""
    from math import gcd

    if a == 0 and b == 0 and c == 0:
        return Fraction(0)
    nums = [a.numerator, b.numerator, c.numerator]
    dens = [a.denominator, b.denominator, c.denominator]
    common_den = 1
    for d in dens:
        common_den = common_den * d // gcd(common_den, d)
    scaled = [n * (common_den // d) for n, d in zip(nums, dens, strict=True)]
    g = 0
    for v in scaled:
        g = gcd(g, abs(v))
    if g == 0:
        return Fraction(0)
    return Fraction(g, common_den)


def _squared_point_line_distance(
    anchor: tuple[Fraction, ...],
    p: tuple[Fraction, ...],
    q: tuple[Fraction, ...],
) -> Fraction:
    """Exact squared distance from ``anchor`` to the line through ``p, q``."""
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    cross = dx * (anchor[1] - p[1]) - dy * (anchor[0] - p[0])
    norm_sq = dx * dx + dy * dy
    return (cross * cross) / norm_sq


def compute_pinned_line_distance_profile(
    request: PinnedLineDistanceRequest,
) -> PinnedLineDistanceResult:
    """Compute the pinned line-distance profile of a point configuration.

    For every unordered pair of configuration points, take the geometric line it
    spans, collapse pairs defining the same line, and report the exact squared
    distance from the anchor to each distinct line together with every source
    pair.  Lines at equal squared distance are grouped into a sorted
    multiplicity partition.
    """
    from itertools import combinations

    from jacobian.math.geometry.exact._models import PinnedLineEntry

    config = request.configuration
    n = len(config.points)
    points = [_to_fraction_point(p) for p in config.points]
    anchor = tuple(c.as_fraction() for c in request.anchor)

    lines: dict[tuple[Fraction, Fraction, Fraction], list[tuple[int, int]]] = {}
    distances: dict[tuple[Fraction, Fraction, Fraction], Fraction] = {}
    for i, j in combinations(range(n), 2):
        coeffs = _canonical_line_coefficients(points[i], points[j])
        lines.setdefault(coeffs, []).append((i, j))
        if coeffs not in distances:
            distances[coeffs] = _squared_point_line_distance(
                anchor, points[i], points[j]
            )

    # Sort distinct lines by (squared distance, coefficients) for determinism.
    ordered = sorted(lines.keys(), key=lambda c: (distances[c], c))
    entries = tuple(
        PinnedLineEntry(
            line_coefficients=tuple(CanonicalRational.from_fraction(v) for v in coeffs),
            squared_distance=CanonicalRational.from_fraction(distances[coeffs]),
            pairs=tuple(lines[coeffs]),
        )
        for coeffs in ordered
    )

    mult: dict[Fraction, int] = {}
    for entry in entries:
        d = entry.squared_distance.as_fraction()
        mult[d] = mult.get(d, 0) + 1
    multiplicities = tuple(
        (CanonicalRational.from_fraction(d), count) for d, count in sorted(mult.items())
    )
    return PinnedLineDistanceResult(
        configuration=config,
        anchor=request.anchor,
        dimension=2,
        point_count=n,
        lines=entries,
        distance_multiplicities=multiplicities,
    )
