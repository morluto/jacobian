"""Axis-aligned square grid hypergraph constructor."""

from __future__ import annotations

from jacobian.math.combinatorics.finite_structures.axis_aligned_square_grid._models import (
    AxisAlignedSquareGridResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

__all__ = ["construct_axis_aligned_square_grid"]


def construct_axis_aligned_square_grid(
    side_length: int,
) -> AxisAlignedSquareGridResult:
    """Construct the 4-uniform hypergraph of axis-aligned squares in [N]^2.

    Vertices are the N^2 grid points (x,y) for x,y in {0,...,N-1}.
    Hyperedges are the sets {(x,y), (x+d,y), (x,y+d), (x+d,y+d)} for
    every d >= 1 with x+d <= N-1 and y+d <= N-1.
    """
    n = side_length

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
