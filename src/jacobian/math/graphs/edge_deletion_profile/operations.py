"""Edge deletion profile kernel using brute-force chromatic number."""

from __future__ import annotations

import time
from itertools import combinations
from math import comb

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_cancelled,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.edge_deletion_profile._models import (
    MAX_DELETION_ORDER,
    DeletionRow,
    EdgeDeletionProfileResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_EDGE_DELETION_PROFILE_WORK = 50_000_000
MAX_RETAINED_LABEL_CHARACTERS = 16_384
_OWNER_DEADLINE_SECONDS = 3600.0


def _require_execution_active(stage: str) -> None:
    if request_cancelled():
        raise OperationExecutionCancelledError(f"request cancelled {stage}")
    execution = current_request_execution()
    if (
        execution is not None
        and execution.deadline is not None
        and time.monotonic() >= execution.deadline
    ):
        raise OperationExecutionTimeoutError(f"request deadline expired {stage}")


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


def _coloring_work_bound(
    graph: SimpleUndirectedGraph,
    deletion_order: int,
    *,
    source_is_bipartite: bool,
) -> int:
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in graph.vertices}
    for left, right in graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components: list[set[str]] = []
    unseen = set(adjacency)
    while unseen:
        start = unseen.pop()
        component = {start}
        stack = [start]
        while stack:
            vertex = stack.pop()
            for neighbor in adjacency[vertex] & unseen:
                unseen.remove(neighbor)
                component.add(neighbor)
                stack.append(neighbor)
        components.append(component)
    total = 0
    # The chromatic kernel initialises its active-vertex set once per
    # profile row, then filters edges into per-component lists only for
    # active components (those containing edges).  Isolates never trigger
    # per-component filtering, so charge the one-time scan separately
    # and multiply only by the active component count.  Edge deletions
    # can split a source component into as many as deletion_order + 1
    # components, so the maximum post-deletion active component count
    # is source_active_components + deletion_order.
    total += len(graph.vertices) + len(graph.edges)
    active_components = [
        c
        for c in components
        if any(left in c and right in c for left, right in graph.edges)
    ]
    max_active_after_deletion = len(active_components) + deletion_order
    total += max_active_after_deletion * (len(graph.vertices) + len(graph.edges))
    for component in components:
        n = len(component)
        edge_count = sum(
            left in component and right in component for left, right in graph.edges
        )
        if not edge_count or not n:
            total += 1
            continue
        complete_edge_count = n * (n - 1) // 2
        complete = edge_count == complete_edge_count
        if complete and deletion_order == 0:
            total += n
            continue
        if complete:
            # Every nonzero deletion from K_n enters the near-complete
            # missing-edge clique-cover kernel.  Each deleted edge can add a
            # branching choice while the kernel assigns all n vertices; use
            # that source-derived envelope rather than the source chromatic
            # shortcut.
            total += n ** (deletion_order + 1)
            continue
        source_missing = complete_edge_count - edge_count
        max_missing = source_missing + deletion_order
        # Near-complete graph: K_n minus a set F of missing edges.
        # The chromatic number equals the minimum clique cover of F,
        # which the kernel computes via a bounded backtracking search.
        # The greedy upper bound gives an initial k, then exhaustive
        # search tries k-1, k-2, ..., each with k^n branching.  The
        # Charge the actual exhaustive search envelope. The kernel tries
        # every cover size below the greedy upper bound and its assignment
        # tree has at most k**n leaves for a k-cover.
        if max_missing <= n:
            if n > 20:
                return MAX_EDGE_DELETION_PROFILE_WORK + 1
            total += sum(k**n for k in range(1, n + 1))
            continue
        if source_is_bipartite:
            total += edge_count * n * 2
            continue
        component_graph = SimpleUndirectedGraph(
            vertices=tuple(component),
            edges=tuple(
                (left, right)
                for left, right in graph.edges
                if left in component and right in component
            ),
        )
        if _is_bipartite(component_graph):
            total += edge_count * n * 2
            continue
        total += n * sum(k**n for k in range(1, n + 1))
    return total


def _admit_edge_deletion_profile(
    graph: SimpleUndirectedGraph,
    deletion_order: int,
) -> bool:
    """Admit native inputs before row enumeration and chromatic searches."""

    if type(deletion_order) is not int or not 0 <= deletion_order <= MAX_DELETION_ORDER:
        raise OperationDomainValidationError(
            location=("deletion_order",),
            code="graph.edge_deletion.order_out_of_range",
            message=(
                f"deletion_order must be an integer between 0 and {MAX_DELETION_ORDER}"
            ),
        )

    edge_count = len(graph.edges)
    if deletion_order > edge_count:
        raise OperationDomainValidationError(
            location=("deletion_order",),
            code="graph.edge_deletion.order_exceeds_edge_count",
            message="deletion_order must not exceed the number of edges",
        )

    retained_label_characters = sum(len(vertex) for vertex in graph.vertices)
    if retained_label_characters > MAX_RETAINED_LABEL_CHARACTERS:
        raise OperationDomainValidationError(
            location=("graph", "vertices"),
            code="graph.edge_deletion.retained_labels_exceed_bound",
            message=(
                "edge-deletion profile source labels exceed the retained-character "
                "bound"
            ),
        )

    row_count = 0
    source_is_bipartite = _is_bipartite(graph)
    coloring_work = _coloring_work_bound(
        graph,
        deletion_order,
        source_is_bipartite=source_is_bipartite,
    )
    per_row_reconstruction_work = 2 * edge_count
    for order in range(deletion_order + 1):
        row_count += comb(edge_count, order)
        if row_count > MAX_EDGE_DELETION_PROFILE_WORK // max(
            coloring_work + per_row_reconstruction_work, 1
        ):
            raise OperationDomainValidationError(
                location=("graph",),
                code="graph.edge_deletion.work_exceeds_bound",
                message="edge-deletion profile search exceeds its exact work bound",
            )

    return source_is_bipartite


__all__ = ["compute_edge_deletion_profile"]


def compute_edge_deletion_profile(
    graph: SimpleUndirectedGraph,
    deletion_order: int,
) -> EdgeDeletionProfileResult:
    """Return the chromatic number of G-F for every edge-deletion set F.

    For each subset F of edges with |F| <= deletion_order, compute the
    chromatic number of the graph after deleting those edges.
    """
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return compute_edge_deletion_profile(graph, deletion_order)
    if execution.deadline is None:
        bind_request_deadline(execution.started_at + _OWNER_DEADLINE_SECONDS)
    _require_execution_active("before admission")
    source_is_bipartite = _admit_edge_deletion_profile(graph, deletion_order)
    edges = list(graph.edges)
    vertices = list(graph.vertices)

    rows: list[DeletionRow] = []
    for order in range(deletion_order + 1):
        _require_execution_active("during profile enumeration")
        for edge_indices in combinations(range(len(edges)), order):
            _require_execution_active("during profile enumeration")
            deleted = set(edge_indices)
            remaining_edges = [edges[i] for i in range(len(edges)) if i not in deleted]
            chromatic = _chromatic_number(
                vertices,
                remaining_edges,
                source_is_bipartite=source_is_bipartite,
            )
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


def _chromatic_number(
    vertices: list[str],
    edges: list[tuple[str, str]],
    *,
    source_is_bipartite: bool = False,
) -> int:
    """Compute the exact chromatic number by brute-force search."""
    _require_execution_active("during chromatic search")
    if source_is_bipartite:
        return 0 if not vertices else 1 if not edges else 2
    if not edges:
        return 0 if not vertices else 1
    adjacency: dict[str, set[str]] = {v: set() for v in vertices}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    active = {vertex for edge in edges for vertex in edge}
    unseen = set(active)
    component_numbers: list[int] = []
    while unseen:
        _require_execution_active("during chromatic search")
        start = unseen.pop()
        component = {start}
        stack = [start]
        while stack:
            vertex = stack.pop()
            for neighbor in adjacency[vertex] & unseen:
                unseen.remove(neighbor)
                component.add(neighbor)
                stack.append(neighbor)
        component_vertices = [vertex for vertex in vertices if vertex in component]
        component_edges = [
            edge for edge in edges if edge[0] in component and edge[1] in component
        ]
        component_adjacency = {
            vertex: adjacency[vertex] & component for vertex in component_vertices
        }
        n = len(component_vertices)
        complete_edge_count = n * (n - 1) // 2
        if len(component_edges) == complete_edge_count:
            component_numbers.append(n)
        elif len(component_edges) < complete_edge_count:
            missing = complete_edge_count - len(component_edges)
            # Near-complete graph: K_n minus a set F of missing edges.
            # The chromatic number equals the minimum clique cover of F,
            # because a colour class is an clique in F (vertices pairwise
            # non-adjacent in K_n - F).  Only use the clique-cover formula
            # when the component is near-complete (missing few edges
            # relative to complete); otherwise the complement is an
            # arbitrary graph and the formula does not apply.
            if missing <= n:
                component_edge_set = set(component_edges)
                component_missing_edges: list[tuple[str, str]] = [
                    (edge[0], edge[1])
                    for edge in (
                        sorted(combo) for combo in combinations(component_vertices, 2)
                    )
                    if (edge[0], edge[1]) not in component_edge_set
                ]
                clique_cover = (
                    n - 1
                    if len(component_missing_edges) == 1
                    else _min_clique_cover(component_vertices, component_missing_edges)
                )
                component_numbers.append(clique_cover)
            elif _is_bipartite_edges(component_vertices, component_edges):
                component_numbers.append(2)
            else:
                for k in range(1, n + 1):
                    _require_execution_active("during chromatic search")
                    if _try_k_color(component_vertices, component_adjacency, k):
                        component_numbers.append(k)
                        break
    return max(component_numbers, default=1)


def _min_clique_cover(vertices: list[str], edges: list[tuple[str, str]]) -> int:
    """Return the minimum number of cliques covering all vertices.

    In the missing-edge graph F, a clique is a set of vertices that are
    pairwise connected by missing edges — equivalently, a set that can
    share one colour in K_n minus F.  The minimum clique cover of F is
    the chromatic number of K_n minus F.

    For the bounded near-complete regime the missing-edge set is tiny,
    so an exhaustive search over partitions is both simple and exact.
    A greedy bound prunes the search: the clique cover number is at
    most n (all singletons) and at least ceil(n / max_clique_size).
    """

    # Build the adjacency of the missing-edge graph.
    adj: dict[str, set[str]] = {v: set() for v in vertices}
    for left, right in edges:
        adj[left].add(right)
        adj[right].add(left)

    # Enumerate all maximal cliques containing each vertex.
    # For small graphs this is fast; for the bounded domain the
    # missing-edge graph has at most n edges on at most n vertices.

    n = len(vertices)

    # Greedy clique partition: repeatedly take the largest clique
    # from the remaining vertices.  This gives an upper bound.
    def _greedy_clique_partition() -> int:
        remaining = list(vertices)
        count = 0
        while remaining:
            # Greedily build the largest clique starting from first vertex.
            clique: list[str] = [remaining[0]]
            for v in remaining[1:]:
                if all(v in adj[c] for c in clique):
                    clique.append(v)
            for v in clique:
                remaining.remove(v)
            count += 1
        return count

    upper = _greedy_clique_partition()

    # Exhaustive search for a better cover using the greedy upper bound.
    # Try partitioning into k cliques for k = 1 to upper.
    def _can_cover(k: int) -> bool:
        # Try to partition vertices into k cliques.
        # Assign vertices one by one to one of k colour classes,
        # ensuring each class is a clique in the missing-edge graph.
        assignment: list[int] = [-1] * n

        def backtrack(idx: int, used: list[set[str]]) -> bool:
            _require_execution_active("during clique cover search")
            if idx == n:
                return True
            v = vertices[idx]
            for c in range(min(k, idx + 1)):
                if all(vertices[j] in adj[v] for j in range(idx) if assignment[j] == c):
                    assignment[idx] = c
                    used[c].add(v)
                    if backtrack(idx + 1, used):
                        return True
                    used[c].discard(v)
                    assignment[idx] = -1
            return False

        return backtrack(0, [set() for _ in range(k)])

    for k in range(1, upper):
        _require_execution_active("during clique cover search")
        if _can_cover(k):
            return k
    return upper


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
        _require_execution_active("during coloring search")
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
