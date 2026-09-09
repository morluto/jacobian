"""Domain functions for graph morphism operations."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian._execution import (
    OperationResourceExhaustedError,
    OperationWorkLedger,
    request_checkpoint,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.morphisms._models import (
    MAX_CYCLE_SEARCH_PATHS,
    MAX_SUBGRAPH_CANDIDATE_CHECKS,
    MORPHISM_MAX_VERTICES,
    FixedLengthCycleResult,
    GraphHomomorphism,
    GraphHomomorphismObstruction,
    GraphVertexMap,
    HomomorphismCheckResult,
    SubgraphPatternFindResult,
    _cycle_source_edges,
    _first_homomorphism_obstruction,
    _validate_cycle_witness,
    _validate_embedding_witness,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = [
    "fixed_length_cycle",
    "homomorphism_check",
    "subgraph_pattern_find",
    "verify_fixed_length_cycle",
    "verify_homomorphism_check",
    "verify_subgraph_pattern_find",
]

MAX_MORPHISM_RETAINED_LABEL_CHARACTERS = 10_000_000
_WITNESS_PRESOLVE_WORK = 1_024


def _graph_label_characters(graph: SimpleUndirectedGraph) -> int:
    return sum(len(vertex) for vertex in graph.vertices) + sum(
        len(left) + len(right) for left, right in graph.edges
    )


def _reject_retained_labels(location: tuple[str, ...]) -> None:
    raise OperationDomainValidationError(
        location=location,
        code="graph.morphism.retained_labels_exceed_bound",
        message="morphism result exceeds the retained label-character bound",
    )


def _admit_cycle_request(graph: SimpleUndirectedGraph, length: int) -> None:
    """Admit the cross-field search and retained-result envelope."""
    if type(length) is not int or length < 3:
        raise OperationDomainValidationError(
            location=("length",),
            code="graph.cycle.length_bound",
            message="cycle length must be an integer of at least 3",
        )
    n = len(graph.vertices)
    if n > MORPHISM_MAX_VERTICES:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.cycle.vertex_bound",
            message=f"graph must have at most {MORPHISM_MAX_VERTICES} vertices",
        )
    if length > n:
        raise OperationDomainValidationError(
            location=("length",),
            code="graph.cycle.length_bound",
            message="cycle length must not exceed the vertex count",
        )
    _admit_cycle_retained(graph, length)


def _admit_complete_cycle_search(adjacency: list[set[int]], length: int) -> None:
    """Admit every charged DFS prefix needed for a negative decision."""

    maximum_degree = max(map(len, adjacency), default=0)
    work = 0
    paths_at_depth = len(adjacency)
    for _ in range(1, length):
        paths_at_depth *= maximum_degree
        work += paths_at_depth
        if work > MAX_CYCLE_SEARCH_PATHS:
            raise OperationResourceAdmissionError(
                location=("length",),
                code="graph.cycle.search_bound",
                message=(
                    "complete fixed-length cycle search exceeds the "
                    f"{MAX_CYCLE_SEARCH_PATHS}-path work budget"
                ),
            )


def _admit_cycle_retained(graph: SimpleUndirectedGraph, length: int) -> None:
    """Admit the source and witness representation without replaying search work."""
    largest_label = max((len(vertex) for vertex in graph.vertices), default=0)
    if (
        _graph_label_characters(graph) + length * largest_label
        > MAX_MORPHISM_RETAINED_LABEL_CHARACTERS
    ):
        _reject_retained_labels(("graph",))


def _admit_homomorphism_request(vertex_map: GraphVertexMap) -> None:
    """Admit one source map and its largest possible retained obstruction."""

    labels = vertex_map.source_graph.vertices + vertex_map.target_graph.vertices
    retained = (
        _graph_label_characters(vertex_map.source_graph)
        + _graph_label_characters(vertex_map.target_graph)
        + sum(
            len(row.source_vertex) + len(row.target_vertex) for row in vertex_map.rows
        )
        + 4 * max((len(label) for label in labels), default=0)
    )
    if retained > MAX_MORPHISM_RETAINED_LABEL_CHARACTERS:
        _reject_retained_labels(("vertex_map",))


def _admit_subgraph_request(
    pattern: SimpleUndirectedGraph, host: SimpleUndirectedGraph
) -> None:
    """Admit the cross-field search and retained-result envelope."""
    pattern_size = len(pattern.vertices)
    if pattern_size > MORPHISM_MAX_VERTICES:
        raise OperationDomainValidationError(
            location=("pattern",),
            code="graph.subgraph.pattern_vertex_bound",
            message=f"pattern must have at most {MORPHISM_MAX_VERTICES} vertices",
        )
    if pattern_size > len(host.vertices):
        raise OperationDomainValidationError(
            location=("pattern",),
            code="graph.subgraph.pattern_size_bound",
            message="pattern must not have more vertices than the host",
        )
    _admit_subgraph_retained(pattern, host)


def _admit_subgraph_retained(
    pattern: SimpleUndirectedGraph, host: SimpleUndirectedGraph
) -> None:
    """Admit the source and embedding representation without replaying search work."""
    pattern_size = len(pattern.vertices)
    largest_host_label = max((len(vertex) for vertex in host.vertices), default=0)
    retained = (
        _graph_label_characters(pattern)
        + _graph_label_characters(host)
        + pattern_size * largest_host_label
    )
    if retained > MAX_MORPHISM_RETAINED_LABEL_CHARACTERS:
        _reject_retained_labels(("pattern", "host"))


def homomorphism_check(vertex_map: GraphVertexMap) -> HomomorphismCheckResult:
    _admit_homomorphism_request(vertex_map)
    obstruction = _first_homomorphism_obstruction(vertex_map)
    if obstruction is None:
        return HomomorphismCheckResult._from_kernel(
            status="HOMOMORPHISM",
            homomorphism=GraphHomomorphism(vertex_map=vertex_map),
        )
    source_edge, image_vertices = obstruction
    return HomomorphismCheckResult._from_kernel(
        status="EDGE_IMAGE_NOT_EDGE",
        obstruction=GraphHomomorphismObstruction(
            vertex_map=vertex_map,
            source_edge=source_edge,
            image_vertices=image_vertices,
        ),
    )


def verify_homomorphism_check(claim: HomomorphismCheckResult) -> bool:
    """Verify a serialized homomorphism decision against its retained map."""
    try:
        if claim.status == "HOMOMORPHISM":
            if claim.homomorphism is None:
                return False
            vertex_map = claim.homomorphism.vertex_map
        else:
            if claim.obstruction is None:
                return False
            vertex_map = claim.obstruction.vertex_map
        return homomorphism_check(vertex_map) == claim
    except (OperationDomainValidationError, TypeError, ValueError):
        return False


def _canonical_label_adjacency(
    vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> tuple[dict[str, int], list[set[int]]]:
    """Map labels to indices and build integer adjacency for search."""
    index = {label: i for i, label in enumerate(vertices)}
    n = len(vertices)
    adj: list[set[int]] = [set() for _ in range(n)]
    for u_label, v_label in edges:
        u = index[u_label]
        v = index[v_label]
        adj[u].add(v)
        adj[v].add(u)
    return index, adj


def _find_cycle_of_length(
    vertices: tuple[str, ...],
    adjacency: list[set[int]],
    length: int,
    *,
    max_search_paths: int = MAX_CYCLE_SEARCH_PATHS,
) -> tuple[int, ...] | None:
    """Return one simple cycle of exactly ``length`` vertices, or ``None``.

    Exhaustive search over vertex indices with a runtime path-count budget.
    """
    adj = adjacency
    n = len(vertices)
    ledger = OperationWorkLedger(max_search_paths)
    request_checkpoint("before fixed-length cycle search")

    # To avoid reporting rotations, fix the smallest vertex of the cycle as the
    # start and restrict every other vertex to be strictly larger than the
    # start; within that, the path may visit any eligible larger neighbor.  The
    # closing edge from the last vertex back to the start completes the cycle.
    path: list[int] = []

    def dfs(start: int, last: int) -> bool:
        if len(path) == length:
            return start in adj[last]
        for nxt in range(start + 1, n):
            if nxt not in adj[last] or nxt in path:
                continue
            ledger.charge()
            if ledger.consumed % 1024 == 0:
                request_checkpoint("during fixed-length cycle search")
            path.append(nxt)
            if dfs(start, nxt):
                return True
            path.pop()
        return False

    for start in range(n):
        path = [start]
        if dfs(start, start):
            return tuple(path)
    return None


def _decide_cycle_indices(
    graph: SimpleUndirectedGraph, length: int
) -> tuple[int, ...] | None:
    """Run a bounded witness presolve, then an admitted complete search."""

    _, adjacency = _canonical_label_adjacency(graph.vertices, graph.edges)
    try:
        return _find_cycle_of_length(
            graph.vertices,
            adjacency,
            length,
            max_search_paths=min(_WITNESS_PRESOLVE_WORK, MAX_CYCLE_SEARCH_PATHS),
        )
    except OperationResourceExhaustedError:
        _admit_complete_cycle_search(adjacency, length)
        return _find_cycle_of_length(
            graph.vertices,
            adjacency,
            length,
            max_search_paths=MAX_CYCLE_SEARCH_PATHS,
        )


def fixed_length_cycle(
    graph: SimpleUndirectedGraph, length: int
) -> FixedLengthCycleResult:
    """Decide whether ``graph`` contains a simple cycle of length ``length``.

    Returns ``EXISTS`` with one ordered cycle witness (a sequence of vertex
    labels whose consecutive vertices and the last-to-first vertex are edges)
    or ``DOES_NOT_EXIST`` after exhaustive bounded search.  The witness cycle is
    a subgraph and may have chords; this is distinct from girth (shortest
    cycle) and from Hamiltonicity (spanning).
    """
    _admit_cycle_request(graph, length)
    k = length
    found = _decide_cycle_indices(graph, k)
    request_checkpoint("before fixed-length cycle result construction")
    if found is not None:
        return FixedLengthCycleResult._from_kernel(
            graph=graph,
            decision="EXISTS",
            length=k,
            cycle=tuple(graph.vertices[i] for i in found),
        )
    return FixedLengthCycleResult._from_kernel(
        graph=graph,
        decision="DOES_NOT_EXIST",
        length=k,
        cycle=(),
    )


def verify_fixed_length_cycle(claim: FixedLengthCycleResult) -> bool:
    """Verify a serialized cycle decision and its optional witness."""
    try:
        vertex_set, edge_set = _cycle_source_edges(claim.graph)
        if claim.decision == "EXISTS":
            _admit_cycle_retained(claim.graph, claim.length)
            _validate_cycle_witness(claim.cycle, claim.length, vertex_set, edge_set)
            return True
        if claim.cycle:
            return False
        _admit_cycle_request(claim.graph, claim.length)
        return _decide_cycle_indices(claim.graph, claim.length) is None
    except (TypeError, ValueError):
        return False


def _candidate_preserves_pattern_edges(
    candidate_idx: int,
    pattern_idx: int,
    pattern_adj: tuple[tuple[int, ...], ...],
    vertex_map_idx: list[int],
    host_adj: tuple[set[int], ...],
) -> bool:
    for neighbor_idx in pattern_adj[pattern_idx]:
        mapped_idx = vertex_map_idx[neighbor_idx]
        if mapped_idx != -1 and mapped_idx not in host_adj[candidate_idx]:
            return False
    return True


def _backtrack_subgraph_embedding(
    position: int,
    pattern_order: tuple[int, ...],
    pattern_adj: tuple[tuple[int, ...], ...],
    candidate_domains: tuple[tuple[int, ...], ...],
    host_adj: tuple[set[int], ...],
    vertex_map_idx: list[int],
    used_host_idx: set[int],
    ledger: OperationWorkLedger,
) -> bool:
    if position == len(pattern_order):
        return True
    pattern_idx = pattern_order[position]
    for host_idx in candidate_domains[pattern_idx]:
        ledger.charge()
        if ledger.consumed % 1024 == 0:
            request_checkpoint("during subgraph-pattern search")
        if host_idx in used_host_idx:
            continue
        if not _candidate_preserves_pattern_edges(
            host_idx, pattern_idx, pattern_adj, vertex_map_idx, host_adj
        ):
            continue
        vertex_map_idx[pattern_idx] = host_idx
        used_host_idx.add(host_idx)
        if _backtrack_subgraph_embedding(
            position + 1,
            pattern_order,
            pattern_adj,
            candidate_domains,
            host_adj,
            vertex_map_idx,
            used_host_idx,
            ledger,
        ):
            return True
        used_host_idx.discard(host_idx)
        vertex_map_idx[pattern_idx] = -1
    return False


@dataclass(frozen=True)
class _SubgraphSearchPlan:
    pattern_order: tuple[int, ...]
    pattern_adjacency: tuple[tuple[int, ...], ...]
    candidate_domains: tuple[tuple[int, ...], ...]
    host_adjacency: tuple[set[int], ...]


def _build_subgraph_search_plan(
    pattern_vertices: tuple[str, ...],
    pattern_edges: tuple[tuple[str, str], ...],
    host_vertices: tuple[str, ...],
    host_edges: tuple[tuple[str, str], ...],
) -> _SubgraphSearchPlan:
    """Build the degree-filtered plan shared by presolve and full search."""

    # Normalize host labels to indices once, as the cycle kernel does: the
    # assignment-count admission bounds search paths, so every per-check
    # cost must be index work rather than label-length-dependent string
    # comparisons, which long shared-prefix labels would multiply into an
    # unbounded admitted run.
    _, host_adj = _canonical_label_adjacency(host_vertices, host_edges)
    p_n = len(pattern_vertices)
    pattern_index = {label: i for i, label in enumerate(pattern_vertices)}
    # Map pattern vertex label -> degree for ordering.
    pattern_degree: dict[str, int] = dict.fromkeys(pattern_vertices, 0)
    for u, v in pattern_edges:
        pattern_degree[u] += 1
        pattern_degree[v] += 1
    pattern_order = tuple(
        sorted(range(p_n), key=lambda i: -pattern_degree[pattern_vertices[i]])
    )
    # Pattern edges as pairs of indices in pattern vertex order.
    pattern_edge_idx = tuple(
        (pattern_index[u], pattern_index[v]) for u, v in pattern_edges
    )
    # Pattern adjacency for quick neighbor checks.
    pattern_adj: list[list[int]] = [[] for _ in range(p_n)]
    for u_idx, v_idx in pattern_edge_idx:
        pattern_adj[u_idx].append(v_idx)
        pattern_adj[v_idx].append(u_idx)
    candidate_domains = tuple(
        tuple(
            host_idx
            for host_idx, neighbors in enumerate(host_adj)
            if len(neighbors) >= len(pattern_adj[pattern_idx])
        )
        for pattern_idx in range(p_n)
    )
    return _SubgraphSearchPlan(
        pattern_order=pattern_order,
        pattern_adjacency=tuple(tuple(row) for row in pattern_adj),
        candidate_domains=candidate_domains,
        host_adjacency=tuple(host_adj),
    )


def _admit_complete_subgraph_search(plan: _SubgraphSearchPlan) -> None:
    """Admit every candidate scan in the degree-filtered search plan."""

    host_size = len(plan.host_adjacency)
    partial_assignments = 1
    work = 0
    for depth, pattern_index in enumerate(plan.pattern_order):
        domain_size = len(plan.candidate_domains[pattern_index])
        work += partial_assignments * domain_size
        if work > MAX_SUBGRAPH_CANDIDATE_CHECKS:
            raise OperationResourceAdmissionError(
                location=("pattern", "host"),
                code="graph.subgraph.search_bound",
                message=(
                    "complete subgraph-pattern search exceeds the "
                    f"{MAX_SUBGRAPH_CANDIDATE_CHECKS}-candidate work budget"
                ),
            )
        partial_assignments *= min(domain_size, host_size - depth)


def _find_subgraph_embedding(
    plan: _SubgraphSearchPlan,
    max_candidate_checks: int = MAX_SUBGRAPH_CANDIDATE_CHECKS,
) -> tuple[int, ...] | None:
    """Return host indices ordered by pattern vertex order, or ``None``."""

    ledger = OperationWorkLedger(max_candidate_checks)
    request_checkpoint("before subgraph-pattern search")
    vertex_map_idx: list[int] = [-1] * len(plan.pattern_order)
    used_host_idx: set[int] = set()

    if _backtrack_subgraph_embedding(
        0,
        plan.pattern_order,
        plan.pattern_adjacency,
        plan.candidate_domains,
        plan.host_adjacency,
        vertex_map_idx,
        used_host_idx,
        ledger,
    ):
        return tuple(vertex_map_idx)
    return None


def _decide_subgraph_embedding(
    pattern: SimpleUndirectedGraph, host: SimpleUndirectedGraph
) -> tuple[int, ...] | None:
    """Run a bounded witness presolve, then an admitted complete search."""

    plan = _build_subgraph_search_plan(
        pattern.vertices, pattern.edges, host.vertices, host.edges
    )
    candidate_union = {
        host_index for domain in plan.candidate_domains for host_index in domain
    }
    if len(candidate_union) < len(plan.pattern_order):
        return None
    try:
        return _find_subgraph_embedding(
            plan,
            max_candidate_checks=min(
                _WITNESS_PRESOLVE_WORK, MAX_SUBGRAPH_CANDIDATE_CHECKS
            ),
        )
    except OperationResourceExhaustedError:
        _admit_complete_subgraph_search(plan)
        return _find_subgraph_embedding(
            plan,
            max_candidate_checks=MAX_SUBGRAPH_CANDIDATE_CHECKS,
        )


def subgraph_pattern_find(
    pattern: SimpleUndirectedGraph, host: SimpleUndirectedGraph
) -> SubgraphPatternFindResult:
    """Find an injective edge-preserving embedding of ``pattern`` in ``host``.

    Ordinary (non-induced) subgraph containment: an injective map from pattern
    vertices to host vertices such that every pattern edge maps to a host edge.
    Returns ``EXISTS`` with one witness vertex map (ordered by pattern vertex
    order) or ``DOES_NOT_EXIST`` after exhaustive bounded search.
    """
    _admit_subgraph_request(pattern, host)
    found = _decide_subgraph_embedding(pattern, host)
    request_checkpoint("before subgraph-pattern result construction")
    if found is not None:
        return SubgraphPatternFindResult._from_kernel(
            pattern=pattern,
            host=host,
            decision="EXISTS",
            vertex_map=tuple(host.vertices[i] for i in found),
        )
    return SubgraphPatternFindResult._from_kernel(
        pattern=pattern,
        host=host,
        decision="DOES_NOT_EXIST",
        vertex_map=(),
    )


def verify_subgraph_pattern_find(claim: SubgraphPatternFindResult) -> bool:
    """Verify a serialized embedding decision and its optional witness."""
    try:
        if claim.decision == "EXISTS":
            _admit_subgraph_retained(claim.pattern, claim.host)
            _validate_embedding_witness(claim.pattern, claim.host, claim.vertex_map)
            return True
        if claim.vertex_map:
            return False
        _admit_subgraph_request(claim.pattern, claim.host)
        return _decide_subgraph_embedding(claim.pattern, claim.host) is None
    except (TypeError, ValueError):
        return False
