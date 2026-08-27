"""Exact divisibility-incidence graph kernel."""

from __future__ import annotations

from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.number_theory._divisibility_graph_models import (
    DivisibilityIncidenceGraphRequest,
    DivisibilityIncidenceGraphResult,
)


def compute_divisibility_incidence_graph(
    request: DivisibilityIncidenceGraphRequest,
) -> DivisibilityIncidenceGraphResult:
    """Build a bipartite simple graph joining l to r iff l divides r.

    Left vertices are labeled 'L{i}' and right vertices 'R{j}'.
    An edge connects L{i} to R{j} exactly when left_family[i] divides right_family[j].
    """
    left = request.left_family
    right = request.right_family

    left_labels = [f"L{i}" for i in range(len(left))]
    right_labels = [f"R{i}" for i in range(len(right))]

    vertices = tuple(left_labels + right_labels)
    edges: list[tuple[str, str]] = []

    left_vals = [int(value) for value in left]
    right_vals = [int(r) for r in right]

    for i, lv in enumerate(left_vals):
        for j, rv in enumerate(right_vals):
            if rv % lv == 0:
                edges.append((f"L{i}", f"R{j}"))

    edges.sort()
    return DivisibilityIncidenceGraphResult(
        left_family=tuple(left),
        right_family=tuple(right),
        graph=SimpleUndirectedGraph(vertices=vertices, edges=tuple(edges)),
    )


__all__ = ["compute_divisibility_incidence_graph"]
