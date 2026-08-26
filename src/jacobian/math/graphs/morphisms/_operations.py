"""Domain functions for graph morphism operations."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from jacobian.math.graphs.morphisms._models import (
    _MAX_SEARCH_PATHS_PER_PASS,
    FixedLengthCycleRequest,
    FixedLengthCycleResult,
    GraphHomomorphism,
    GraphHomomorphismObstruction,
    HomomorphismCheckRequest,
    HomomorphismCheckResult,
    SubgraphPatternFindRequest,
    SubgraphPatternFindResult,
    _cycle_source_edges,
    _first_homomorphism_obstruction,
    _require_negative_cycle_domain,
    _require_negative_embedding_domain,
    _validate_cycle_witness,
    _validate_embedding_witness,
)


def compute_homomorphism_check(
    request: HomomorphismCheckRequest,
) -> HomomorphismCheckResult:
    obstruction = _first_homomorphism_obstruction(request.vertex_map)
    if obstruction is None:
        return HomomorphismCheckResult._from_kernel(
            status="HOMOMORPHISM",
            homomorphism=GraphHomomorphism(vertex_map=request.vertex_map),
        )
    source_edge, image_vertices = obstruction
    return HomomorphismCheckResult._from_kernel(
        status="EDGE_IMAGE_NOT_EDGE",
        obstruction=GraphHomomorphismObstruction(
            vertex_map=request.vertex_map,
            source_edge=source_edge,
            image_vertices=image_vertices,
        ),
    )


def verify_homomorphism_check_result(result: HomomorphismCheckResult) -> bool:
    """Independently verify a source-bound homomorphism-check claim."""

    if result.status == "HOMOMORPHISM":
        return (
            result.homomorphism is not None
            and result.obstruction is None
            and _first_homomorphism_obstruction(result.homomorphism.vertex_map) is None
        )
    if result.homomorphism is not None or result.obstruction is None:
        return False
    expected = _first_homomorphism_obstruction(result.obstruction.vertex_map)
    return expected == (
        result.obstruction.source_edge,
        result.obstruction.image_vertices,
    )


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


def find_cycle_of_length(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    length: int,
) -> tuple[int, ...] | None:
    """Return one simple cycle of exactly ``length`` vertices, or ``None``.

    Exhaustive bounded search over vertex indices.  Callers must enforce
    the request's path-count admission before invoking this kernel so the
    search terminates inside a tested budget.
    """
    _, adj = _canonical_label_adjacency(vertices, edges)
    n = len(vertices)

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


def compute_fixed_length_cycle(
    request: FixedLengthCycleRequest,
) -> FixedLengthCycleResult:
    """Decide whether ``graph`` contains a simple cycle of length ``length``.

    Returns ``EXISTS`` with one ordered cycle witness (a sequence of vertex
    labels whose consecutive vertices and the last-to-first vertex are edges)
    or ``DOES_NOT_EXIST`` after exhaustive bounded search.  The witness cycle is
    a subgraph and may have chords; this is distinct from girth (shortest
    cycle) and from Hamiltonicity (spanning).
    """
    graph = request.graph
    k = request.length
    found = find_cycle_of_length(graph.vertices, graph.edges, k)
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


class SearchBudgetExceededError(RuntimeError):
    """The bounded search exhausted its candidate-check budget.

    A search stopped at its budget establishes nothing: it is neither a
    witness nor a negative decision, so callers must surface the typed
    non-conclusion instead of projecting it into a decision.
    """


def verify_fixed_length_cycle_result(result: FixedLengthCycleResult) -> bool:
    """Independently check a cycle decision inside its admitted envelope."""

    if result.decision == "EXISTS":
        try:
            vertices, edges = _cycle_source_edges(result.graph)
            _validate_cycle_witness(result.cycle, result.length, vertices, edges)
        except PydanticCustomError:
            return False
        return True
    if result.cycle:
        return False
    try:
        _require_negative_cycle_domain(result.graph, result.length)
    except PydanticCustomError:
        return False
    return (
        find_cycle_of_length(result.graph.vertices, result.graph.edges, result.length)
        is None
    )


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
    host_adj: list[set[int]],
    host_count: int,
    vertex_map_idx: list[int],
    used_host_idx: set[int],
    budget: int | None,
    candidate_checks: list[int],
) -> bool:
    if position == len(pattern_order):
        return True
    pattern_idx = pattern_order[position]
    for host_idx in range(host_count):
        if budget is not None:
            candidate_checks[0] += 1
            if candidate_checks[0] > budget:
                raise SearchBudgetExceededError(
                    f"subgraph-pattern search exceeded its "
                    f"{budget}-candidate-check per-pass budget"
                )
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
            host_adj,
            host_count,
            vertex_map_idx,
            used_host_idx,
            budget,
            candidate_checks,
        ):
            return True
        used_host_idx.discard(host_idx)
        vertex_map_idx[pattern_idx] = -1
    return False


def find_subgraph_embedding(
    pattern_vertices: tuple[str, ...],
    pattern_edges: tuple[tuple[str, str], ...],
    host_vertices: tuple[str, ...],
    host_edges: tuple[tuple[str, str], ...],
    max_candidate_checks: int | None = None,
) -> tuple[int, ...] | None:
    """Return host indices ordered by pattern vertex order, or ``None``.

    Ordinary (non-induced) subgraph containment via exhaustive bounded
    backtracking.  Callers must enforce the request's assignment-count
    admission before invoking this kernel.  Every host-candidate scan at an
    internal backtracking node is charged against ``max_candidate_checks``
    when given; exceeding it raises ``SearchBudgetExceededError`` so a
    partially searched space can never masquerade as a negative decision.
    """
    # Normalize host labels to indices once, as the cycle kernel does: the
    # assignment-count admission bounds search paths, so every per-check
    # cost must be index work rather than label-length-dependent string
    # comparisons, which long shared-prefix labels would multiply into an
    # unbounded admitted run.
    _, host_adj = _canonical_label_adjacency(host_vertices, host_edges)
    p_n = len(pattern_vertices)
    h_n = len(host_vertices)
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
    candidate_checks = [0]
    # Every admitted request declares this bound; explicit verification uses
    # the same charged accounting as execution.
    budget = max_candidate_checks if max_candidate_checks is not None else None

    vertex_map_idx: list[int] = [-1] * p_n  # pattern idx -> host idx
    used_host_idx: set[int] = set()
    # Pattern adjacency for quick neighbor checks.
    pattern_adj: list[list[int]] = [[] for _ in range(p_n)]
    for u_idx, v_idx in pattern_edge_idx:
        pattern_adj[u_idx].append(v_idx)
        pattern_adj[v_idx].append(u_idx)

    if _backtrack_subgraph_embedding(
        0,
        pattern_order,
        pattern_adj,
        host_adj,
        h_n,
        vertex_map_idx,
        used_host_idx,
        budget,
        candidate_checks,
    ):
        return tuple(vertex_map_idx[i] for i in range(p_n))
    return None


def compute_subgraph_pattern_find(
    request: SubgraphPatternFindRequest,
) -> SubgraphPatternFindResult:
    """Find an injective edge-preserving embedding of ``pattern`` in ``host``.

    Ordinary (non-induced) subgraph containment: an injective map from pattern
    vertices to host vertices such that every pattern edge maps to a host edge.
    Returns ``EXISTS`` with one witness vertex map (ordered by pattern vertex
    order) or ``DOES_NOT_EXIST`` after exhaustive bounded search.
    """
    try:
        found = find_subgraph_embedding(
            request.pattern.vertices,
            request.pattern.edges,
            request.host.vertices,
            request.host.edges,
            max_candidate_checks=_MAX_SEARCH_PATHS_PER_PASS,
        )
    except SearchBudgetExceededError:
        # A search stopped at its budget establishes nothing either way.
        return SubgraphPatternFindResult._from_kernel(
            pattern=request.pattern,
            host=request.host,
            decision="BUDGET_EXCEEDED",
            vertex_map=(),
        )
    if found is not None:
        return SubgraphPatternFindResult._from_kernel(
            pattern=request.pattern,
            host=request.host,
            decision="EXISTS",
            vertex_map=tuple(request.host.vertices[i] for i in found),
        )
    return SubgraphPatternFindResult._from_kernel(
        pattern=request.pattern,
        host=request.host,
        decision="DOES_NOT_EXIST",
        vertex_map=(),
    )


def verify_subgraph_pattern_find_result(result: SubgraphPatternFindResult) -> bool:
    """Independently check a bounded subgraph-containment claim."""

    if result.decision == "EXISTS":
        try:
            _validate_embedding_witness(result.pattern, result.host, result.vertex_map)
        except PydanticCustomError:
            return False
        return True
    if result.vertex_map:
        return False
    if result.decision == "DOES_NOT_EXIST":
        try:
            _require_negative_embedding_domain(result.pattern, result.host)
        except PydanticCustomError:
            return False
    try:
        found = find_subgraph_embedding(
            result.pattern.vertices,
            result.pattern.edges,
            result.host.vertices,
            result.host.edges,
            max_candidate_checks=_MAX_SEARCH_PATHS_PER_PASS,
        )
    except SearchBudgetExceededError:
        return result.decision == "BUDGET_EXCEEDED"
    if result.decision == "BUDGET_EXCEEDED":
        return False
    return found is None
