"""Domain-owned exact geometry operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.exact._models import (
    CircumradiusMultiplicityEntry,
    CircumradiusProfileEntry,
    CircumradiusProfileRequest,
    CircumradiusProfileResult,
    CircumradiusTripleDisposition,
    DistanceGraphRequest,
    DistanceGraphResult,
    DistanceMultiplicityEntry,
    DistanceProfileRequest,
    DistanceProfileResult,
    LabelledRationalPoint,
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



def _squared_circumradius(
    p: tuple[Fraction, ...],
    q: tuple[Fraction, ...],
    r: tuple[Fraction, ...],
) -> tuple[CircumradiusTripleDisposition, Fraction | None]:
    """Return the disposition and exact squared circumradius of one triple.

    For side squared-lengths ``a2, b2, c2`` the squared circumradius is
    ``R^2 = (a2 b2 c2) / (16 K^2)`` where
    ``16 K^2 = 2(a2 b2 + b2 c2 + c2 a2) - (a2^2 + b2^2 + c2^2)`` is the
    Heron expression in squared sides.  A zero denominator means the three
    points are collinear, so no circumcircle exists.
    """
    a2 = _squared_distance(q, r)
    b2 = _squared_distance(p, r)
    c2 = _squared_distance(p, q)
    sixteen_k_squared = (
        2 * (a2 * b2 + b2 * c2 + c2 * a2) - (a2 * a2 + b2 * b2 + c2 * c2)
    )
    if sixteen_k_squared == 0:
        return CircumradiusTripleDisposition.DEGENERATE, None
    radius_squared = (a2 * b2 * c2) / sixteen_k_squared
    return CircumradiusTripleDisposition.NONDEGENERATE, radius_squared


def compute_circumradius_profile(
    request: CircumradiusProfileRequest,
) -> CircumradiusProfileResult:
    """Compute the exact squared circumradius of every unordered triple.

    Each triple receives an explicit mathematical disposition: a nondegenerate
    triangle with its exact squared circumradius, or a degenerate (collinear)
    triple for which no circumcircle exists.  Triples sharing each radius are
    grouped into a sorted multiplicity profile so collisions are directly
    inspectable.
    """
    from collections import Counter
    from itertools import combinations

    config = request.configuration
    n = len(config.points)
    points = [_to_fraction_point(p) for p in config.points]

    triples: list[CircumradiusProfileEntry] = []
    radii: Counter[Fraction] = Counter()
    for i, j, k in combinations(range(n), 3):
        disposition, radius = _squared_circumradius(points[i], points[j], points[k])
        squared_radius_value: CanonicalRational | None = None
        if disposition is CircumradiusTripleDisposition.NONDEGENERATE:
            assert radius is not None
            squared_radius_value = CanonicalRational.from_fraction(radius)
            radii[radius] += 1
        triples.append(
            CircumradiusProfileEntry(
                triple=(i, j, k),
                disposition=disposition,
                squared_radius=squared_radius_value,
            ),
        )

    nondegenerate = sum(radii.values())
    multiplicities = tuple(
        CircumradiusMultiplicityEntry(
            squared_radius=CanonicalRational.from_fraction(radius),
            triple_count=count,
        )
        for radius, count in sorted(radii.items())
    )
    return CircumradiusProfileResult(
        dimension=2,
        point_count=n,
        triples=tuple(triples),
        multiplicities=multiplicities,
        nondegenerate_count=nondegenerate,
        degenerate_count=len(triples) - nondegenerate,
    )


__all__ = [
    "compute_circumradius_profile",
    "compute_distance_graph",
    "compute_distance_profile",
]
