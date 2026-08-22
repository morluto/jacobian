"""Domain-owned exact geometry operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.exact._models import (
    CollinearTriplesRequest,
    ConcyclicQuadruplesRequest,
    DistanceGraphRequest,
    DistanceGraphResult,
    DistanceMultiplicityEntry,
    DistanceProfileRequest,
    DistanceProfileResult,
    IncidenceSearchResult,
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


__all__ = [
    "compute_collinear_triples",
    "compute_concyclic_quadruples",
    "compute_distance_graph",
    "compute_distance_profile",
]


def _det3(rows: tuple[tuple[Fraction, Fraction, Fraction], ...]) -> Fraction:
    a, b, c = rows
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _det4(rows: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    """Exact 4x4 determinant by cofactor expansion along the first row."""
    n = 4

    def minor(
        matrix: tuple[tuple[Fraction, ...], ...], drop_col: int
    ) -> tuple[tuple[Fraction, ...], ...]:
        return tuple(
            tuple(matrix[r][c] for c in range(n) if c != drop_col) for r in range(1, n)
        )

    def det3x3(m: tuple[tuple[Fraction, ...], ...]) -> Fraction:
        a, b, c = m
        return (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )

    total = Fraction(0)
    for col in range(n):
        sign = Fraction(1) if col % 2 == 0 else Fraction(-1)
        total += sign * rows[0][col] * det3x3(minor(rows, col))
    return total


def _are_collinear(
    p: tuple[Fraction, ...],
    q: tuple[Fraction, ...],
    r: tuple[Fraction, ...],
) -> bool:
    return (
        _det3(
            (
                (p[0], p[1], Fraction(1)),
                (q[0], q[1], Fraction(1)),
                (r[0], r[1], Fraction(1)),
            ),
        )
        == 0
    )


def _are_concyclic(
    p: tuple[Fraction, ...],
    q: tuple[Fraction, ...],
    r: tuple[Fraction, ...],
    s: tuple[Fraction, ...],
) -> bool:
    rows = tuple(
        (pt[0], pt[1], pt[0] * pt[0] + pt[1] * pt[1], Fraction(1))
        for pt in (p, q, r, s)
    )
    return _det4(rows) == 0


def compute_collinear_triples(
    request: CollinearTriplesRequest,
) -> IncidenceSearchResult:
    """Find a witness collinear triple, or establish none exists."""
    from itertools import combinations

    config = request.configuration
    n = len(config.points)
    points = [_to_fraction_point(p) for p in config.points]
    witnesses: list[tuple[int, int, int]] = []
    for i, j, k in combinations(range(n), 3):
        if _are_collinear(points[i], points[j], points[k]):
            witnesses.append((i, j, k))
    return IncidenceSearchResult(
        configuration=config,
        dimension=2,
        point_count=n,
        holds=bool(witnesses),
        witnesses=tuple(witnesses),
        kind="COLLINEAR_TRIPLE",
    )


def compute_concyclic_quadruples(
    request: ConcyclicQuadruplesRequest,
) -> IncidenceSearchResult:
    """Find a witness concyclic quadruple, or establish none exists."""
    from itertools import combinations

    config = request.configuration
    n = len(config.points)
    points = [_to_fraction_point(p) for p in config.points]
    witnesses: list[tuple[int, int, int, int]] = []
    for i, j, k, ell in combinations(range(n), 4):
        # A quadruple containing a collinear triple is degenerate: it lies on
        # no proper circle (infinite radius), so exclude it from witnesses.
        if any(
            _are_collinear(points[a], points[b], points[c])
            for a, b, c in ((i, j, k), (i, j, ell), (i, k, ell), (j, k, ell))
        ):
            continue
        if _are_concyclic(points[i], points[j], points[k], points[ell]):
            witnesses.append((i, j, k, ell))
    return IncidenceSearchResult(
        configuration=config,
        dimension=2,
        point_count=n,
        holds=bool(witnesses),
        witnesses=tuple(witnesses),
        kind="CONCYCLIC_QUADRUPLE",
    )
