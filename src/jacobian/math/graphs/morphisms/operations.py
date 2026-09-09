"""Domain functions for graph morphism operations."""

from __future__ import annotations

from jacobian._execution import OperationWorkLedger, request_checkpoint
from jacobian.catalog.models import OperationDomainValidationError
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


def _canonical_max_degree(graph: SimpleUndirectedGraph) -> int:
    """Maximum vertex degree of the canonical source graph.

    Owner-local admission helper: the cycle work estimate must be computed
    from canonical values without replaying the search kernel.
    """
    degree: dict[str, int] = dict.fromkeys(graph.vertices, 0)
    for u, v in graph.edges:
        degree[u] += 1
        degree[v] += 1
    return max(degree.values(), default=0)


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
    # Conservative full-negative-case admission: worst-case DFS work for a
    # k-cycle is bounded by the number of simple directed paths of length
    # k-1, at most n*(d_max)^(k-1). Every accepted request must terminate
    # inside the tested bound even when no witness exists, so this estimate
    # is enforced before enumeration begins; the runtime ledger remains only
    # as defense-in-depth and never as the primary admission.
    work = n * (_canonical_max_degree(graph) ** (length - 1))
    if work > MAX_CYCLE_SEARCH_PATHS:
        raise OperationDomainValidationError(
            location=("length",),
            code="graph.cycle.search_bound",
            message=(
                "fixed-length cycle search exceeds the "
                f"{MAX_CYCLE_SEARCH_PATHS}-path work budget"
            ),
        )
    _admit_cycle_retained(graph, length)


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
    # Conservative full-negative-case admission: worst-case backtracking work
    # is bounded by the falling factorial of host vertices taken
    # pattern-at-a-time. A negative decision requires the complete search, so
    # requests whose exhaustive family exceeds the candidate budget are
    # rejected before enumeration begins; the runtime ledger remains only as
    # defense-in-depth.
    assignments = 1
    for step in range(pattern_size):
        assignments *= len(host.vertices) - step
        if assignments > MAX_SUBGRAPH_CANDIDATE_CHECKS:
            raise OperationDomainValidationError(
                location=("pattern", "host"),
                code="graph.subgraph.search_bound",
                message=(
                    "subgraph-pattern search exceeds the "
                    f"{MAX_SUBGRAPH_CANDIDATE_CHECKS}-assignment work budget"
                ),
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
    edges: tuple[tuple[str, str], ...],
    length: int,
    *,
    max_search_paths: int = MAX_CYCLE_SEARCH_PATHS,
) -> tuple[int, ...] | None:
    """Return one simple cycle of exactly ``length`` vertices, or ``None``.

    Exhaustive bounded search over vertex indices.  Callers must enforce
    the request's path-count admission before invoking this kernel so the
    search terminates inside a tested budget.
    """
    _, adj = _canonical_label_adjacency(vertices, edges)
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
    found = _find_cycle_of_length(
        graph.vertices, graph.edges, k, max_search_paths=MAX_CYCLE_SEARCH_PATHS
    )
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
        return (
            _find_cycle_of_length(
                claim.graph.vertices,
                claim.graph.edges,
                claim.length,
                max_search_paths=MAX_CYCLE_SEARCH_PATHS,
            )
            is None
        )
    except (TypeError, ValueError):
        return False


def _candidate_preserves_pattern_edges(
    candidate_idx: int,
    pattern_idx: int,
    pattern_adj: list[list[int]],
    vertex_map_idx: list[int],
    host_adj: list[set[int]],
) -> bool:
    for neighbor_idx in pattern_adj[pattern_idx]:
        mapped_idx = vertex_map_idx[neighbor_idx]
        if mapped_idx != -1 and mapped_idx not in host_adj[candidate_idx]:
            return False
    return True


def _backtrack_subgraph_embedding(
    position: int,
    pattern_order: list[int],
    pattern_adj: list[list[int]],
    candidate_domains: list[tuple[int, ...]],
    host_adj: list[set[int]],
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


def _find_subgraph_embedding(
    pattern_vertices: tuple[str, ...],
    pattern_edges: tuple[tuple[str, str], ...],
    host_vertices: tuple[str, ...],
    host_edges: tuple[tuple[str, str], ...],
    max_candidate_checks: int = MAX_SUBGRAPH_CANDIDATE_CHECKS,
) -> tuple[int, ...] | None:
    """Return host indices ordered by pattern vertex order, or ``None``.

    Ordinary (non-induced) subgraph containment via bounded backtracking.
    Every host-candidate scan at an internal backtracking node is charged
    against ``max_candidate_checks`` before it executes. Exhausting that
    allowance raises an execution error, so a partial search can never
    masquerade as a negative decision.
    """
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
    pattern_order = sorted(
        range(p_n), key=lambda i: -pattern_degree[pattern_vertices[i]]
    )
    # Pattern edges as pairs of indices in pattern vertex order.
    pattern_edge_idx = tuple(
        (pattern_index[u], pattern_index[v]) for u, v in pattern_edges
    )
    ledger = OperationWorkLedger(max_candidate_checks)
    request_checkpoint("before subgraph-pattern search")

    vertex_map_idx: list[int] = [-1] * p_n  # pattern idx -> host idx
    used_host_idx: set[int] = set()
    # Pattern adjacency for quick neighbor checks.
    pattern_adj: list[list[int]] = [[] for _ in range(p_n)]
    for u_idx, v_idx in pattern_edge_idx:
        pattern_adj[u_idx].append(v_idx)
        pattern_adj[v_idx].append(u_idx)
    candidate_domains = [
        tuple(
            host_idx
            for host_idx, neighbors in enumerate(host_adj)
            if len(neighbors) >= len(pattern_adj[pattern_idx])
        )
        for pattern_idx in range(p_n)
    ]

    if _backtrack_subgraph_embedding(
        0,
        pattern_order,
        pattern_adj,
        candidate_domains,
        host_adj,
        vertex_map_idx,
        used_host_idx,
        ledger,
    ):
        return tuple(vertex_map_idx[i] for i in range(p_n))
    return None


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
    found = _find_subgraph_embedding(
        pattern.vertices,
        pattern.edges,
        host.vertices,
        host.edges,
        max_candidate_checks=MAX_SUBGRAPH_CANDIDATE_CHECKS,
    )
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
        return (
            _find_subgraph_embedding(
                claim.pattern.vertices,
                claim.pattern.edges,
                claim.host.vertices,
                claim.host.edges,
                max_candidate_checks=MAX_SUBGRAPH_CANDIDATE_CHECKS,
            )
            is None
        )
    except (TypeError, ValueError):
        return False
