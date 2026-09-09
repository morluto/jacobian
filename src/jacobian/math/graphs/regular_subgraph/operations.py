"""Find a nonempty k-regular subgraph by bounded edge-subset enumeration."""

from __future__ import annotations

from collections import deque
from itertools import combinations

from jacobian._execution import OperationWorkLedger, request_checkpoint
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.regular_subgraph._models import (
    RegularSubgraphResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

# The previous 16-edge envelope could visit every edge in every subset:
# 16 * 2**15 selected-edge visits. Runtime accounting keeps that ceiling while
# allowing an early witness in a larger source graph.
MAX_REGULAR_SUBGRAPH_WORK = 16 * (1 << 15)


def _trivial_result(
    graph: SimpleUndirectedGraph,
    k: int,
    edges: list[tuple[str, str]],
) -> RegularSubgraphResult | None:
    """Return a constant-work result for k=0 or k=1 when available."""

    if k == 0 and graph.vertices:
        return RegularSubgraphResult(
            graph=graph,
            k=k,
            found=True,
            vertices=(graph.vertices[0],),
            edges=(),
        )
    if k == 1 and edges:
        left_label, right_label = edges[0]
        edge = (
            (left_label, right_label)
            if left_label <= right_label
            else (right_label, left_label)
        )
        return RegularSubgraphResult(
            graph=graph,
            k=k,
            found=True,
            vertices=edge,
            edges=(edge,),
        )
    return None


def _find_cycle_edge_indices(
    edge_pairs: list[tuple[int, int]],
    vertex_count: int,
    ledger: OperationWorkLedger,
) -> tuple[int, ...] | None:
    """Return one cycle's edge indices using an incrementally built forest."""

    parents = list(range(vertex_count))
    ranks = [0] * vertex_count
    forest: list[list[tuple[int, int]]] = [[] for _ in range(vertex_count)]

    def root(vertex: int) -> int:
        while parents[vertex] != vertex:
            parents[vertex] = parents[parents[vertex]]
            vertex = parents[vertex]
        return vertex

    for edge_index, (left, right) in enumerate(edge_pairs):
        ledger.charge()
        if ledger.consumed % 1024 == 0:
            request_checkpoint("during 2-regular cycle search")
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            if ranks[left_root] < ranks[right_root]:
                left_root, right_root = right_root, left_root
            parents[right_root] = left_root
            if ranks[left_root] == ranks[right_root]:
                ranks[left_root] += 1
            forest[left].append((right, edge_index))
            forest[right].append((left, edge_index))
            continue

        previous: dict[int, tuple[int, int] | None] = {left: None}
        pending = deque([left])
        while pending:
            vertex = pending.popleft()
            if vertex == right:
                break
            for neighbor, forest_edge_index in forest[vertex]:
                ledger.charge()
                if ledger.consumed % 1024 == 0:
                    request_checkpoint("during 2-regular cycle reconstruction")
                if neighbor not in previous:
                    previous[neighbor] = (vertex, forest_edge_index)
                    pending.append(neighbor)

        path_edges: list[int] = []
        vertex = right
        previous_step = previous[vertex]
        while previous_step is not None:
            vertex, forest_edge_index = previous_step
            path_edges.append(forest_edge_index)
            previous_step = previous[vertex]
        path_edges.append(edge_index)
        return tuple(path_edges)
    return None


def find_k_regular_subgraph(
    graph: SimpleUndirectedGraph,
    k: int,
) -> RegularSubgraphResult:
    """Return a nonempty k-regular subgraph (vertex set and edge set) or found=false.

    A subgraph is k-regular when every *used* vertex has degree exactly k in the
    selected edge set. The k=2 regime uses exact cycle detection; larger k uses
    bounded edge-subset enumeration. For k=0, any single vertex suffices.
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

    # Trivial witnesses do not require enumerating edge subsets.
    trivial_result = _trivial_result(graph, k, edges)
    if trivial_result is not None:
        request_checkpoint("before k-regular subgraph result construction")
        return trivial_result

    vertex_to_idx = {v: i for i, v in enumerate(vertices)}

    # Precompute edge endpoints as index pairs.
    edge_pairs: list[tuple[int, int]] = []
    for left_label, right_label in edges:
        edge_pairs.append((vertex_to_idx[left_label], vertex_to_idx[right_label]))

    ledger = OperationWorkLedger(MAX_REGULAR_SUBGRAPH_WORK)
    request_checkpoint("before k-regular subgraph search")
    if k == 2:
        cycle_edges = _find_cycle_edge_indices(edge_pairs, n_vertices, ledger)
        request_checkpoint("before k-regular subgraph result construction")
        if cycle_edges is None:
            return RegularSubgraphResult(graph=graph, k=k, found=False)
        selected_labels = [edges[index] for index in cycle_edges]
        used_labels = tuple(
            sorted({label for edge in selected_labels for label in edge})
        )
        return RegularSubgraphResult(
            graph=graph,
            k=k,
            found=True,
            vertices=used_labels,
            edges=tuple(sorted(selected_labels)),
        )

    # Try edge subsets in increasing size. A nonempty subgraph with at least one
    # vertex of positive degree requires at least k+1 vertices and ceil(k*|V|/2) edges.
    min_edges_needed = (k + 1) * k // 2 if k > 0 else 0

    for edge_count in range(max(1, min_edges_needed), n_edges + 1):
        for edge_combo in combinations(range(n_edges), edge_count):
            ledger.charge(edge_count)
            if ledger.consumed % 1024 == 0:
                request_checkpoint("during k-regular subgraph search")
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
                request_checkpoint("before k-regular subgraph result construction")
                return RegularSubgraphResult(
                    graph=graph,
                    k=k,
                    found=True,
                    vertices=used_vertex_labels,
                    edges=tuple(sorted(used_edge_list)),
                )

    request_checkpoint("before k-regular subgraph result construction")
    return RegularSubgraphResult(graph=graph, k=k, found=False)


def verify_k_regular_subgraph(claim: RegularSubgraphResult) -> bool:
    """Return whether a retained witness or absence claim is valid."""
    if not claim.found:
        try:
            return not find_k_regular_subgraph(claim.graph, claim.k).found
        except OperationDomainValidationError:
            return False
    vertices = set(claim.vertices)
    if not vertices or len(vertices) != len(claim.vertices):
        return False
    if not vertices <= set(claim.graph.vertices):
        return False
    graph_edges = {frozenset(edge) for edge in claim.graph.edges}
    if any(frozenset(edge) not in graph_edges for edge in claim.edges):
        return False
    degrees = dict.fromkeys(vertices, 0)
    for left, right in claim.edges:
        if left not in vertices or right not in vertices:
            return False
        degrees[left] += 1
        degrees[right] += 1
    return all(degree == claim.k for degree in degrees.values())


__all__ = ["find_k_regular_subgraph", "verify_k_regular_subgraph"]
