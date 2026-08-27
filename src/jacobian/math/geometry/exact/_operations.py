"""Domain-owned exact geometry operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.exact._line_arithmetic import (
    canonical_line_coefficients,
    squared_point_line_distance,
)
from jacobian.math.geometry.exact._models import (
    DistanceGraphRequest,
    DistanceMultiplicityEntry,
    DistanceProfileRequest,
    DistanceProfileResult,
    LabelledRationalPoint,
    PinnedLineDistanceRequest,
    PinnedLineDistanceResult,
)
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph


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
) -> IndexedSimpleUndirectedGraph:
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

    return IndexedSimpleUndirectedGraph(
        vertex_count=n,
        edges=tuple(edges),
    )


__all__ = [
    "compute_distance_graph",
    "compute_distance_profile",
    "compute_pinned_line_distance_profile",
]


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
        coeffs = canonical_line_coefficients(points[i], points[j])
        lines.setdefault(coeffs, []).append((i, j))
        if coeffs not in distances:
            distances[coeffs] = squared_point_line_distance(
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
    return PinnedLineDistanceResult._from_kernel(
        request,
        lines=entries,
        distance_multiplicities=multiplicities,
    )
