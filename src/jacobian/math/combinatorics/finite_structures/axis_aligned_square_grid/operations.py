"""Axis-aligned square grid hypergraph constructor."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.axis_aligned_square_grid._models import (
    MAX_SIDE_LENGTH,
    AxisAlignedSquareGridResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    FiniteHypergraph,
)

__all__ = ["construct_axis_aligned_square_grid"]

def _admit_side_length(side_length: int) -> int:
    if isinstance(side_length, bool) or not isinstance(side_length, int):
        raise OperationDomainValidationError(
            location=("side_length",),
            code="square_grid.invalid_side_length",
            message="side_length must be a strict integer",
        )
    if not 1 <= side_length <= MAX_SIDE_LENGTH:
        raise OperationDomainValidationError(
            location=("side_length",),
            code="square_grid.side_length_bound",
            message=f"side_length must be between 1 and {MAX_SIDE_LENGTH}",
        )
    n = side_length
    edge_count = n * (n - 1) * (2 * n - 1) // 6
    incidence_count = 4 * edge_count
    if edge_count > MAX_EDGES:
        raise OperationDomainValidationError(
            location=("side_length",),
            code="square_grid.edge_bound",
            message=f"the grid would contain {edge_count} edges, over the {MAX_EDGES}-edge bound",
        )
    if incidence_count > MAX_TOTAL_INCIDENCES:
        raise OperationDomainValidationError(
            location=("side_length",),
            code="square_grid.incidence_bound",
            message=(
                f"the grid would contain {incidence_count} incidences, over the "
                f"{MAX_TOTAL_INCIDENCES}-incidence bound"
            ),
        )

    return n


def construct_axis_aligned_square_grid(
    side_length: int,
) -> AxisAlignedSquareGridResult:
    """Construct the 4-uniform hypergraph of axis-aligned squares in [N]^2.

    Vertices are the N^2 grid points (x,y) for x,y in {0,...,N-1}.
    Hyperedges are the sets {(x,y), (x+d,y), (x,y+d), (x+d,y+d)} for
    every d >= 1 with x+d <= N-1 and y+d <= N-1.
    """
    n = _admit_side_length(side_length)

    def vertex_label(x: int, y: int) -> str:
        return f"({x},{y})"

    vertices = tuple(vertex_label(x, y) for y in range(n) for x in range(n))

    edges: list[tuple[str, tuple[str, ...]]] = []
    edge_index = 0
    for y in range(n):
        for x in range(n):
            for d in range(1, n):
                if x + d <= n - 1 and y + d <= n - 1:
                    edge = (
                        vertex_label(x, y),
                        vertex_label(x + d, y),
                        vertex_label(x, y + d),
                        vertex_label(x + d, y + d),
                    )
                    edges.append((f"square_{edge_index}", tuple(sorted(edge))))
                    edge_index += 1

    hypergraph = FiniteHypergraph(
        vertices=vertices,
        edges=tuple(edges),
    )
    return AxisAlignedSquareGridResult(
        side_length=n,
        hypergraph=hypergraph,
    )
