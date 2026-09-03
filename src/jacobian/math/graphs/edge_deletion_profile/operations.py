"""Edge-deletion profile kernel with bounded exact coloring algorithms."""

from __future__ import annotations

import time
from itertools import combinations
from math import comb

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
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
    request_checkpoint(stage)
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
        source_missing = complete_edge_count - edge_count
        max_missing = source_missing + deletion_order
        # Near-complete graph: K_n minus a set F of missing edges.
        # The chromatic number equals the minimum clique cover of F,
        # whose nontrivial part is supported only on vertices incident to a
        # missing edge. Each deletion can add at most two such vertices.
        if max_missing <= n:
            source_missing_support = sum(
                len(adjacency[vertex] & component) < n - 1 for vertex in component
            )
            support = min(n, source_missing_support + 2 * deletion_order)
            total += n * n + 2**support + 3**support
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
        # The general exact kernel uses subset dynamic programming.  It
        # classifies every vertex subset once and then enumerates only
        # independent colour classes containing one fixed pivot.  Across all
        # states this visits fewer than 2**n + 3**n bounded bit-mask states.
        total += n * n + 2**n + 3**n
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
    edge_keys = {frozenset(edge) for edge in edges}
    source_missing_edges = [
        edge for edge in combinations(vertices, 2) if frozenset(edge) not in edge_keys
    ]
    complement_profile = (
        source_missing_edges
        if len(source_missing_edges) + deletion_order <= len(vertices)
        else None
    )

    rows: list[DeletionRow] = []
    for order in range(deletion_order + 1):
        _require_execution_active("during profile enumeration")
        for edge_indices in combinations(range(len(edges)), order):
            _require_execution_active("during profile enumeration")
            if complement_profile is not None:
                missing_edges = [
                    *complement_profile,
                    *(edges[index] for index in edge_indices),
                ]
                chromatic = _min_clique_cover(vertices, missing_edges)
            else:
                deleted = set(edge_indices)
                remaining_edges = [
                    edge for index, edge in enumerate(edges) if index not in deleted
                ]
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
    """Compute the exact chromatic number with structural and subset kernels."""
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
                component_numbers.append(
                    _chromatic_number_subset_dp(
                        component_vertices,
                        component_adjacency,
                    )
                )
    return max(component_numbers, default=1)


def _min_clique_cover(vertices: list[str], edges: list[tuple[str, str]]) -> int:
    """Return the minimum number of cliques covering all vertices.

    In the missing-edge graph F, a clique is a set of vertices that are
    pairwise connected by missing edges — equivalently, a set that can
    share one colour in K_n minus F.  The minimum clique cover of F is
    the chromatic number of K_n minus F.

    Vertices outside the support of the missing edges are isolated in F and
    therefore contribute one singleton clique each. The remaining exact
    partition is computed only on the missing-edge support.
    """

    adj: dict[str, set[str]] = {v: set() for v in vertices}
    for left, right in edges:
        adj[left].add(right)
        adj[right].add(left)

    active_vertices = [vertex for vertex in vertices if adj[vertex]]
    inactive_count = len(vertices) - len(active_vertices)
    vertex_indices = {vertex: index for index, vertex in enumerate(active_vertices)}
    compatibility_masks = [
        sum(1 << vertex_indices[neighbor] for neighbor in adj[vertex])
        for vertex in active_vertices
    ]
    return inactive_count + _minimum_compatible_partition(compatibility_masks)


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


def _chromatic_number_subset_dp(
    vertices: list[str], adjacency: dict[str, set[str]]
) -> int:
    """Return the exact chromatic number by partitioning into independent sets."""

    order = len(vertices)
    if order == 0:
        return 0

    vertex_indices = {vertex: index for index, vertex in enumerate(vertices)}
    adjacency_masks = [
        sum(1 << vertex_indices[neighbor] for neighbor in adjacency[vertex])
        for vertex in vertices
    ]
    full_mask = (1 << order) - 1
    compatibility_masks = [
        full_mask ^ (1 << index) ^ adjacency_masks[index] for index in range(order)
    ]
    return _minimum_compatible_partition(compatibility_masks)


def _minimum_compatible_partition(compatibility_masks: list[int]) -> int:
    """Partition vertices into the fewest pairwise-compatible classes."""

    order = len(compatibility_masks)
    state_count = 1 << order

    compatible = [False] * state_count
    compatible[0] = True
    for mask in range(1, state_count):
        _require_execution_active("during compatible-subset classification")
        pivot = mask & -mask
        pivot_index = pivot.bit_length() - 1
        remainder = mask ^ pivot
        compatible[mask] = compatible[remainder] and (
            remainder & ~compatibility_masks[pivot_index] == 0
        )

    partition_counts = [order + 1] * state_count
    partition_counts[0] = 0
    for mask in range(1, state_count):
        _require_execution_active("during subset partition search")
        pivot = mask & -mask
        remainder = mask ^ pivot
        submask = remainder
        while True:
            partition_class = submask | pivot
            if compatible[partition_class]:
                partition_counts[mask] = min(
                    partition_counts[mask],
                    partition_counts[mask ^ partition_class] + 1,
                )
            if submask == 0:
                break
            submask = (submask - 1) & remainder

    return partition_counts[-1]
