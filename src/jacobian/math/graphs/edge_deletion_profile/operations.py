"""Edge deletion profile kernel using brute-force chromatic number."""

from __future__ import annotations

from itertools import combinations
from math import comb

from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.edge_deletion_profile._models import (
    MAX_DELETION_ORDER,
    DeletionRow,
    EdgeDeletionProfileResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_EDGE_DELETION_PROFILE_WORK = 50_000_000


def _json_array_size(item_size: int, count: int) -> int:
    return 2 + max(count - 1, 0) + item_size * count


def _is_bipartite(graph: SimpleUndirectedGraph) -> bool:
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in graph.vertices}
    for left, right in graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    colors: dict[str, bool] = {}
    for start in graph.vertices:
        if start in colors:
            continue
        colors[start] = False
        stack = [start]
        while stack:
            vertex = stack.pop()
            for neighbor in adjacency[vertex]:
                if neighbor not in colors:
                    colors[neighbor] = not colors[vertex]
                    stack.append(neighbor)
                elif colors[neighbor] == colors[vertex]:
                    return False
    return True


def _coloring_work_bound(graph: SimpleUndirectedGraph, deletion_order: int) -> int:
    n = len(graph.vertices)
    edge_count = len(graph.edges)
    if not edge_count or not n:
        return 1
    complete = edge_count == n * (n - 1) // 2
    if complete and deletion_order == 0:
        return n
    if _is_bipartite(graph):
        return edge_count * n * 2
    # The kernel tries each k-colouring in turn. Charge the complete finite
    # search tree rather than the graph's edge count alone.
    return n * sum(k**n for k in range(1, n + 1))


def _admit_edge_deletion_profile(
    graph: SimpleUndirectedGraph,
    deletion_order: int,
) -> None:
    """Admit native inputs before row enumeration and chromatic searches."""

    if type(deletion_order) is not int or not 0 <= deletion_order <= MAX_DELETION_ORDER:
        raise OperationDomainValidationError(
            location=("deletion_order",),
            code="graph.edge_deletion.order_out_of_range",
            message=(
                f"deletion_order must be an integer between 0 and {MAX_DELETION_ORDER}"
            ),
        )

    vertex_count = len(graph.vertices)
    edge_count = len(graph.edges)
    if deletion_order > edge_count:
        raise OperationDomainValidationError(
            location=("deletion_order",),
            code="graph.edge_deletion.order_exceeds_edge_count",
            message="deletion_order must not exceed the number of edges",
        )

    row_count = 0
    coloring_work = _coloring_work_bound(graph, deletion_order)
    for order in range(deletion_order + 1):
        row_count += comb(edge_count, order)
        if row_count > MAX_EDGE_DELETION_PROFILE_WORK // max(coloring_work, 1):
            raise OperationDomainValidationError(
                location=("graph",),
                code="graph.edge_deletion.work_exceeds_bound",
                message="edge-deletion profile search exceeds its exact work bound",
            )

    try:
        graph_bytes = len(encode_strict_json(graph.model_dump(mode="json")))
    except CanonicalizationError as exc:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.edge_deletion.result_exceeds_output_bound",
            message="edge-deletion profile result exceeds the canonical output bound",
        ) from exc
    index_bytes = max(1, len(str(max(edge_count - 1, 0))))
    row_bytes = strict_json_object_size(
        (
            (
                "deleted_edge_indices",
                _json_array_size(index_bytes, deletion_order),
            ),
            ("chromatic_number", max(1, len(str(vertex_count)))),
        )
    )
    rows_bytes = _json_array_size(row_bytes, row_count)
    result_bytes = strict_json_object_size(
        (
            ("graph", graph_bytes),
            ("deletion_order", max(1, len(str(deletion_order)))),
            ("rows", rows_bytes),
        )
    )
    if result_bytes > CanonicalLimits().max_output_bytes:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.edge_deletion.result_exceeds_output_bound",
            message="edge-deletion profile result exceeds the canonical output bound",
        )


__all__ = ["compute_edge_deletion_profile"]


def compute_edge_deletion_profile(
    graph: SimpleUndirectedGraph,
    deletion_order: int,
) -> EdgeDeletionProfileResult:
    """Return the chromatic number of G-F for every edge-deletion set F.

    For each subset F of edges with |F| <= deletion_order, compute the
    chromatic number of the graph after deleting those edges.
    """
    _admit_edge_deletion_profile(graph, deletion_order)
    edges = list(graph.edges)
    vertices = list(graph.vertices)

    rows: list[DeletionRow] = []
    for order in range(deletion_order + 1):
        for edge_indices in combinations(range(len(edges)), order):
            deleted = set(edge_indices)
            remaining_edges = [edges[i] for i in range(len(edges)) if i not in deleted]
            chromatic = _chromatic_number(vertices, remaining_edges)
            rows.append(
                DeletionRow(
                    deleted_edge_indices=tuple(edge_indices),
                    chromatic_number=chromatic,
                )
            )

    return EdgeDeletionProfileResult(
        graph=graph,
        deletion_order=deletion_order,
        rows=tuple(rows),
    )


def _chromatic_number(vertices: list[str], edges: list[tuple[str, str]]) -> int:
    """Compute the exact chromatic number by brute-force search."""
    n = len(vertices)
    if n == 0:
        return 0
    if not edges:
        return 1
    if len(edges) == n * (n - 1) // 2:
        return n
    if _is_bipartite_edges(vertices, edges):
        return 2

    adjacency: dict[str, set[str]] = {v: set() for v in vertices}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    for k in range(1, n + 1):
        if _try_k_color(vertices, adjacency, k):
            return k
    return n


def _is_bipartite_edges(vertices: list[str], edges: list[tuple[str, str]]) -> bool:
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in vertices}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    colors: dict[str, bool] = {}
    for start in vertices:
        if start in colors:
            continue
        colors[start] = False
        stack = [start]
        while stack:
            vertex = stack.pop()
            for neighbor in adjacency[vertex]:
                if neighbor not in colors:
                    colors[neighbor] = not colors[vertex]
                    stack.append(neighbor)
                elif colors[neighbor] == colors[vertex]:
                    return False
    return True


def _try_k_color(vertices: list[str], adjacency: dict[str, set[str]], k: int) -> bool:
    """Check if the graph is k-colorable."""
    colors: dict[str, int] = {}

    def backtrack(idx: int) -> bool:
        if idx == len(vertices):
            return True
        v = vertices[idx]
        for c in range(k):
            if all(colors.get(n, -1) != c for n in adjacency[v]):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                del colors[v]
        return False

    return backtrack(0)
