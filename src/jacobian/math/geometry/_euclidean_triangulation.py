"""Certified bounded Euclidean triangulation of strict convex rational polygons."""

from __future__ import annotations

from fractions import Fraction

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry._models import (
    EUCLIDEAN_TRIANGULATION_COMPARISON_PRECISION_BITS,
    EuclideanComparisonUnresolved,
    EuclideanConvexPolygonTriangulationRequest,
    EuclideanConvexPolygonTriangulationResult,
    EuclideanDiagonal,
    EuclideanLengthExpression,
    EuclideanTriangulationSplitEntry,
    PolygonTriangle,
    _compare_euclidean_root_sums,
    _euclidean_squared_length,
    _require_euclidean_triangulation_envelope,
)


def _admit_euclidean_triangulation(
    request: EuclideanConvexPolygonTriangulationRequest,
) -> tuple[tuple[Fraction, Fraction], ...]:
    try:
        return _require_euclidean_triangulation_envelope(request.polygon)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("polygon", "points"), code=exc.type, message=exc.message()
        ) from exc


def _expression(values: tuple[Fraction, ...]) -> EuclideanLengthExpression:
    return EuclideanLengthExpression(
        squared_lengths=tuple(
            CanonicalRational.from_fraction(value) for value in sorted(values)
        )
    )


def minimum_euclidean_weight_triangulation(
    request: EuclideanConvexPolygonTriangulationRequest,
) -> EuclideanConvexPolygonTriangulationResult:
    """Find one exact Euclidean minimum, only after every DP choice is certified.

    The dynamic program charges the boundary ``(start, end)`` of each
    non-root subproblem.  Thus each selected non-hull diagonal is represented
    exactly once.  Its cost remains an exact square-root expression; Arb is
    used privately only for rigorously deciding comparisons between finite
    expressions at the declared precision.
    """

    points = _admit_euclidean_triangulation(request)
    count = len(points)

    def is_hull_edge(first: int, second: int) -> bool:
        return second == first + 1 or (first, second) == (0, count - 1)

    optimum: dict[tuple[int, int], tuple[Fraction, ...]] = {
        (index, index + 1): () for index in range(count - 1)
    }
    split: dict[tuple[int, int], int] = {}
    ledger: list[EuclideanTriangulationSplitEntry] = []
    for span in range(2, count):
        for start in range(count - span):
            end = start + span
            boundary = (
                ()
                if is_hull_edge(start, end)
                else (_euclidean_squared_length(points, start, end),)
            )
            chosen_pivot = start + 1
            chosen = tuple(
                sorted(
                    optimum[start, chosen_pivot] + optimum[chosen_pivot, end] + boundary
                )
            )
            for pivot in range(start + 2, end):
                candidate = tuple(
                    sorted(optimum[start, pivot] + optimum[pivot, end] + boundary)
                )
                order = _compare_euclidean_root_sums(candidate, chosen)
                if order is None:
                    return EuclideanConvexPolygonTriangulationResult._from_kernel(
                        request,
                        status="COMPARISON_UNRESOLVED",
                        unresolved_comparison=EuclideanComparisonUnresolved(
                            start=start,
                            end=end,
                            left_split=pivot,
                            right_split=chosen_pivot,
                            left=_expression(candidate),
                            right=_expression(chosen),
                            precision_bits=EUCLIDEAN_TRIANGULATION_COMPARISON_PRECISION_BITS,
                        ),
                    )
                if order < 0:
                    chosen = candidate
                    chosen_pivot = pivot
            optimum[start, end] = chosen
            split[start, end] = chosen_pivot
            ledger.append(
                EuclideanTriangulationSplitEntry(
                    start=start,
                    end=end,
                    split=chosen_pivot,
                    optimum=_expression(chosen),
                )
            )

    triangles: list[PolygonTriangle] = []
    diagonals: set[tuple[int, int]] = set()

    def reconstruct(start: int, end: int) -> None:
        if end == start + 1:
            return
        pivot = split[start, end]
        triangles.append(PolygonTriangle(vertices=(start, pivot, end)))
        if not is_hull_edge(start, end):
            diagonals.add((start, end))
        reconstruct(start, pivot)
        reconstruct(pivot, end)

    reconstruct(0, count - 1)
    value = optimum[0, count - 1]
    return EuclideanConvexPolygonTriangulationResult._from_kernel(
        request,
        status="CERTIFIED_OPTIMUM",
        diagonals=tuple(
            EuclideanDiagonal(
                first=first,
                second=second,
                squared_length=CanonicalRational.from_fraction(
                    _euclidean_squared_length(points, first, second)
                ),
            )
            for first, second in sorted(diagonals)
        ),
        triangles=tuple(sorted(triangles, key=lambda item: item.vertices)),
        split_table=tuple(ledger),
        optimum=_expression(value),
    )


__all__ = ["minimum_euclidean_weight_triangulation"]
