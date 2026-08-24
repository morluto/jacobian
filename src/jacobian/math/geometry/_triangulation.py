"""Exact bounded rational-weight triangulation of strict convex polygons."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.geometry._models import (
    ConvexPolygonTriangulationRequest,
    ConvexPolygonTriangulationResult,
    PolygonTriangle,
    TriangulationSplitEntry,
    WeightedPolygonDiagonal,
    _reconstruct_split_triangulation,
    _triangulation_subproblem_costs,
)


def _wire(value: Fraction) -> CanonicalRational:
    return CanonicalRational(
        num=format_canonical_integer(value.numerator),
        den=format_canonical_integer(value.denominator),
    )


def minimum_weight_triangulation(
    request: ConvexPolygonTriangulationRequest,
) -> ConvexPolygonTriangulationResult:
    count = len(request.polygon.points)
    weights = {
        (item.first, item.second): item.weight.as_fraction()
        for item in request.diagonal_weights
    }

    def edge_weight(first: int, second: int) -> Fraction:
        pair = (first, second) if first < second else (second, first)
        if second == first + 1 or pair == (0, count - 1):
            return Fraction()
        return weights[pair]

    optimum, split = _triangulation_subproblem_costs(count, edge_weight)
    ledger: list[TriangulationSplitEntry] = []
    for span in range(2, count):
        for start in range(count - span):
            end = start + span
            ledger.append(
                TriangulationSplitEntry(
                    start=start,
                    end=end,
                    split=split[start, end],
                    optimum=_wire(optimum[start, end]),
                )
            )

    diagonal_pairs, triangle_vertices = _reconstruct_split_triangulation(count, split)
    return ConvexPolygonTriangulationResult(
        vertex_count=count,
        diagonals=tuple(
            WeightedPolygonDiagonal(
                first=first, second=second, weight=_wire(weights[pair])
            )
            for pair in diagonal_pairs
            for first, second in (pair,)
        ),
        triangles=tuple(PolygonTriangle(vertices=item) for item in triangle_vertices),
        split_table=tuple(ledger),
        optimum=_wire(optimum[0, count - 1]),
    )


__all__ = ["minimum_weight_triangulation"]
