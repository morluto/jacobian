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

    optimum: dict[tuple[int, int], Fraction] = {
        (index, index + 1): Fraction() for index in range(count - 1)
    }
    split: dict[tuple[int, int], int] = {}
    ledger: list[TriangulationSplitEntry] = []
    for span in range(2, count):
        for start in range(count - span):
            end = start + span
            candidates = [
                (
                    optimum[start, pivot]
                    + optimum[pivot, end]
                    + edge_weight(start, pivot)
                    + edge_weight(pivot, end),
                    pivot,
                )
                for pivot in range(start + 1, end)
            ]
            value, pivot = min(candidates)
            optimum[start, end] = value
            split[start, end] = pivot
            ledger.append(
                TriangulationSplitEntry(
                    start=start,
                    end=end,
                    split=pivot,
                    optimum=_wire(value),
                )
            )

    triangles: list[PolygonTriangle] = []
    diagonals: set[tuple[int, int]] = set()

    def reconstruct(start: int, end: int) -> None:
        if end == start + 1:
            return
        pivot = split[start, end]
        triangles.append(PolygonTriangle(vertices=(start, pivot, end)))
        for pair in ((start, pivot), (pivot, end)):
            ordered = pair if pair[0] < pair[1] else (pair[1], pair[0])
            if pair[1] != pair[0] + 1 and ordered != (0, count - 1):
                diagonals.add(ordered)
        reconstruct(start, pivot)
        reconstruct(pivot, end)

    reconstruct(0, count - 1)
    return ConvexPolygonTriangulationResult(
        vertex_count=count,
        diagonals=tuple(
            WeightedPolygonDiagonal(
                first=first, second=second, weight=_wire(weights[pair])
            )
            for pair in sorted(diagonals)
            for first, second in (pair,)
        ),
        triangles=tuple(sorted(triangles, key=lambda item: item.vertices)),
        split_table=tuple(ledger),
        optimum=_wire(optimum[0, count - 1]),
    )


__all__ = ["minimum_weight_triangulation"]
