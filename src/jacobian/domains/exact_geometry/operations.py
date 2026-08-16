"""Domain adapter for exact geometry operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian.contracts.exact_geometry import (
    DistanceGraphRequest,
    DistanceGraphResult,
    DistanceMultiplicityEntry,
    DistanceProfileRequest,
    DistanceProfileResult,
)


def _to_fraction_point(point) -> tuple[Fraction, ...]:
    return tuple(c.as_fraction() for c in point.coordinates)


def _squared_distance(
    p: tuple[Fraction, ...],
    q: tuple[Fraction, ...],
) -> Fraction:
    return sum((a - b) ** 2 for a, b in zip(p, q, strict=True))


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
            squared_distance=str(d),
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
    target = Fraction(request.target_squared_distance)

    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if _squared_distance(points[i], points[j]) == target:
                edges.append((i, j))

    return DistanceGraphResult(
        vertex_count=n,
        edges=tuple(edges),
    )


__all__ = ["compute_distance_graph", "compute_distance_profile"]
