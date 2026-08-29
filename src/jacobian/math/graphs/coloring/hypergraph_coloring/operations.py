"""Hypergraph colouring decision kernel."""

from __future__ import annotations

from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.graphs.coloring.hypergraph_coloring._models import (
    HypergraphColoringResult,
)

__all__ = ["decide_hypergraph_coloring"]


def decide_hypergraph_coloring(
    hypergraph: FiniteHypergraph,
    palette_size: int,
) -> HypergraphColoringResult:
    """Decide whether the hypergraph has a proper q-colouring.

    A proper q-colouring assigns one of q colours to each vertex such that
    no hyperedge is monochromatic.
    """
    vertices = hypergraph.vertices
    n = len(vertices)
    vertex_index = {v: i for i, v in enumerate(vertices)}

    edges = [tuple(vertex_index[m] for m in members) for _, members in hypergraph.edges]

    if not edges or palette_size < 1:
        if palette_size < 1:
            return HypergraphColoringResult(
                hypergraph=hypergraph,
                palette_size=palette_size,
                colorable=False,
            )
        return HypergraphColoringResult(
            hypergraph=hypergraph,
            palette_size=palette_size,
            colorable=True,
            coloring=tuple(0 for _ in range(n)),
        )

    colors = [0] * n

    def is_edge_safe(edge: tuple[int, ...]) -> bool:
        assigned = [colors[v] for v in edge if colors[v] >= 0]
        if len(assigned) < len(edge):
            return True  # Not all assigned yet
        return len(set(assigned)) > 1

    def backtrack(idx: int) -> list[int] | None:
        if idx == n:
            for edge in edges:
                if not is_edge_safe(edge):
                    return None
            return list(colors)

        for c in range(palette_size):
            colors[idx] = c
            ok = True
            for edge in edges:
                if idx in edge:
                    assigned = [colors[v] for v in edge if colors[v] >= 0]
                    if len(assigned) == len(edge) and len(set(assigned)) == 1:
                        ok = False
                        break
            if ok:
                result = backtrack(idx + 1)
                if result is not None:
                    return result
            colors[idx] = -1
        colors[idx] = 0
        return None

    colors = [-1] * n
    result_colors = backtrack(0)

    if result_colors is not None:
        return HypergraphColoringResult(
            hypergraph=hypergraph,
            palette_size=palette_size,
            colorable=True,
            coloring=tuple(result_colors),
        )
    return HypergraphColoringResult(
        hypergraph=hypergraph,
        palette_size=palette_size,
        colorable=False,
    )
