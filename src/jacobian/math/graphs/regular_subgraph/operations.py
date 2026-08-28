"""Find a nonempty k-regular subgraph by bounded edge-subset enumeration."""

from __future__ import annotations

from itertools import combinations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.regular_subgraph._models import (
    RegularSubgraphResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def find_k_regular_subgraph(
    graph: SimpleUndirectedGraph,
    k: int,
) -> RegularSubgraphResult:
    """Return a nonempty k-regular subgraph (vertex set and edge set) or found=false.

    A subgraph is k-regular when every *used* vertex has degree exactly k in the
    selected edge set. We enumerate edge subsets in increasing size order and
    return the first feasible solution. For k=0, any single vertex suffices.
    """

    if k < 0:
        raise OperationDomainValidationError(
            location=("k",),
            code="graphs.regular_subgraph.negative_k",
            message="k must be nonnegative",
        )
    vertices = graph.vertices
    n_vertices = len(vertices)
    if k > 0 and k >= n_vertices:
        raise OperationDomainValidationError(
            location=("k",),
            code="graphs.regular_subgraph.k_too_large",
            message="k must be less than the number of vertices",
        )
    edges = list(graph.edges)
    n_edges = len(edges)

    vertex_to_idx = {v: i for i, v in enumerate(vertices)}

    # k=0: any single vertex with no edges is a 0-regular subgraph.
    if k == 0 and n_vertices > 0:
        return RegularSubgraphResult(
            graph=graph,
            k=k,
            found=True,
            vertices=(vertices[0],),
            edges=(),
        )

    # Precompute edge endpoints as index pairs.
    edge_pairs: list[tuple[int, int]] = []
    for left_label, right_label in edges:
        edge_pairs.append((vertex_to_idx[left_label], vertex_to_idx[right_label]))

    # Try edge subsets in increasing size. A nonempty subgraph with at least one
    # vertex of positive degree requires at least k+1 vertices and ceil(k*|V|/2) edges.
    min_edges_needed = (k + 1) * k // 2 if k > 0 else 0

    for edge_count in range(max(1, min_edges_needed), n_edges + 1):
        for edge_combo in combinations(range(n_edges), edge_count):
            selected_edges = [edge_pairs[i] for i in edge_combo]
            used_vertices: set[int] = set()
            for left_idx, right_idx in selected_edges:
                used_vertices.add(left_idx)
                used_vertices.add(right_idx)

            if not used_vertices:
                continue

            # Check k-regularity: every used vertex has degree exactly k.
            degree: dict[int, int] = {}
            for left_idx, right_idx in selected_edges:
                degree[left_idx] = degree.get(left_idx, 0) + 1
                degree[right_idx] = degree.get(right_idx, 0) + 1

            if all(d == k for d in degree.values()):
                # Found a k-regular subgraph.
                used_vertex_labels = tuple(sorted(vertices[i] for i in used_vertices))
                used_edge_list: list[tuple[str, str]] = []
                for left_idx, right_idx in selected_edges:
                    left_label, right_label = vertices[left_idx], vertices[right_idx]
                    used_edge_list.append(
                        (left_label, right_label)
                        if left_label <= right_label
                        else (right_label, left_label)
                    )
                return RegularSubgraphResult(
                    graph=graph,
                    k=k,
                    found=True,
                    vertices=used_vertex_labels,
                    edges=tuple(sorted(used_edge_list)),
                )

    return RegularSubgraphResult(graph=graph, k=k, found=False)


__all__ = ["find_k_regular_subgraph"]
