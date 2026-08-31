"""Hypergraph colouring decision kernel."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.graphs.coloring.hypergraph_coloring._models import (
    HypergraphColoringResult,
    _hypergraph_coloring_admission_error,
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
    failure = _hypergraph_coloring_admission_error(hypergraph, palette_size)
    if failure is not None:
        code, message = failure
        raise OperationDomainValidationError(
            location=("hypergraph", "palette_size"),
            code=f"hypergraph_coloring.{code}",
            message=message,
        )
    vertices = hypergraph.vertices
    n = len(vertices)
    vertex_index = {v: i for i, v in enumerate(vertices)}

    edges = [tuple(vertex_index[m] for m in members) for _, members in hypergraph.edges]

    if any(len(edge) <= 1 for edge in edges):
        return HypergraphColoringResult(
            hypergraph=hypergraph,
            palette_size=palette_size,
            colorable=False,
        )
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
    if palette_size == 1:
        return HypergraphColoringResult(
            hypergraph=hypergraph, palette_size=palette_size, colorable=False
        )
    if palette_size >= n:
        return HypergraphColoringResult(
            hypergraph=hypergraph,
            palette_size=palette_size,
            colorable=True,
            coloring=tuple(range(n)),
        )

    result_colors = _find_coloring(n, edges, palette_size)

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


def _find_coloring(
    vertex_count: int,
    edges: list[tuple[int, ...]],
    palette_size: int,
) -> list[int] | None:
    colors = [-1] * vertex_count
    incident_edges: list[list[tuple[int, ...]]] = [[] for _ in range(vertex_count)]
    for edge in edges:
        for vertex in edge:
            incident_edges[vertex].append(edge)

    def backtrack(idx: int) -> list[int] | None:
        if idx == vertex_count:
            return list(colors)

        for color in range(palette_size):
            colors[idx] = color
            if all(_edge_is_safe(edge, colors) for edge in incident_edges[idx]):
                result = backtrack(idx + 1)
                if result is not None:
                    return result
        colors[idx] = -1
        return None

    return backtrack(0)


def _edge_is_safe(edge: tuple[int, ...], colors: list[int]) -> bool:
    assigned = [colors[vertex] for vertex in edge if colors[vertex] >= 0]
    return len(assigned) < len(edge) or len(set(assigned)) > 1
