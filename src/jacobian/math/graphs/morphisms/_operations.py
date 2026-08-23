"""Domain functions for graph morphism operations."""

from __future__ import annotations

from jacobian.math.graphs.morphisms._models import (
    CoreCheckRequest,
    CoreCheckResult,
    FixedLengthCycleRequest,
    FixedLengthCycleResult,
    HomomorphismCheckRequest,
    HomomorphismCheckResult,
    HomomorphismFindRequest,
    HomomorphismFindResult,
    RetractionCheckRequest,
    RetractionCheckResult,
    SubgraphPatternFindRequest,
    SubgraphPatternFindResult,
)


def _adjacency(graph_edges: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    """Return a set of all directed edges (both directions)."""
    adj: set[tuple[int, int]] = set()
    for u, v in graph_edges:
        adj.add((u, v))
        adj.add((v, u))
    return adj


def _is_homomorphism(
    source_edges: tuple[tuple[int, int], ...],
    target_edges: tuple[tuple[int, int], ...],
    vertex_map: list[int],
) -> bool:
    target_adj = _adjacency(target_edges)
    return all((vertex_map[u], vertex_map[v]) in target_adj for u, v in source_edges)


def compute_homomorphism_check(
    request: HomomorphismCheckRequest,
) -> HomomorphismCheckResult:
    is_h = _is_homomorphism(
        request.source_graph.edges,
        request.target_graph.edges,
        list(request.vertex_map),
    )
    return HomomorphismCheckResult(is_homomorphism=is_h)


def compute_homomorphism_find(
    request: HomomorphismFindRequest,
) -> HomomorphismFindResult:
    source = request.source_graph
    target = request.target_graph
    target_adj = _adjacency(target.edges)

    vertex_map: list[int] = [-1] * source.vertex_count

    def backtrack(pos: int) -> bool:
        if pos == source.vertex_count:
            return True
        for candidate in range(target.vertex_count):
            vertex_map[pos] = candidate
            ok = True
            for u, v in source.edges:
                if (
                    u == pos
                    and vertex_map[v] != -1
                    and (vertex_map[u], vertex_map[v]) not in target_adj
                ):
                    ok = False
                    break
                if (
                    v == pos
                    and vertex_map[u] != -1
                    and (vertex_map[u], vertex_map[v]) not in target_adj
                ):
                    ok = False
                    break
            if ok and backtrack(pos + 1):
                return True
            vertex_map[pos] = -1
        return False

    found = backtrack(0)
    return HomomorphismFindResult(
        found=found,
        vertex_map=tuple(vertex_map) if found else (),
    )


def _is_endomorphism(
    source_edges: tuple[tuple[int, int], ...],
    source_adj: set[tuple[int, int]],
    mapping: list[int],
) -> bool:
    return all((mapping[u], mapping[v]) in source_adj for u, v in source_edges)


def compute_core_check(request: CoreCheckRequest) -> CoreCheckResult:
    """A graph is a core iff it has no non-injective endomorphism."""
    source = request.graph
    source_adj = _adjacency(source.edges)
    vertex_map: list[int] = [-1] * source.vertex_count
    has_non_injective = [False]

    def search_non_injective(pos: int) -> bool:
        if pos == source.vertex_count:
            used = set(vertex_map)
            if len(used) < source.vertex_count:
                has_non_injective[0] = True
                return True
            return False
        for candidate in range(source.vertex_count):
            vertex_map[pos] = candidate
            assigned_edges = tuple(
                (u, v)
                for u, v in source.edges
                if vertex_map[u] != -1 and vertex_map[v] != -1
            )
            if _is_endomorphism(
                assigned_edges, source_adj, vertex_map
            ) and search_non_injective(pos + 1):
                return True
            vertex_map[pos] = -1
        return False

    found = search_non_injective(0)
    return CoreCheckResult(is_core=not found)


def compute_retraction_check(
    request: RetractionCheckRequest,
) -> RetractionCheckResult:
    """Check if a retraction onto an induced subgraph exists."""
    source = request.graph
    subgraph = set(request.subgraph_vertices)

    target_edges = [(u, v) for u, v in source.edges if u in subgraph and v in subgraph]
    target_adj = _adjacency(tuple(target_edges))

    vertex_map: list[int] = [-1] * source.vertex_count
    subgraph_list = list(subgraph)

    for v in subgraph_list:
        vertex_map[v] = v

    remaining = [i for i in range(source.vertex_count) if i not in subgraph]

    def backtrack(pos: int) -> bool:
        if pos == len(remaining):
            return True
        v = remaining[pos]
        for candidate in subgraph_list:
            vertex_map[v] = candidate
            ok = True
            for u, w in source.edges:
                if (
                    u == v
                    and vertex_map[w] != -1
                    and (candidate, vertex_map[w]) not in target_adj
                ):
                    ok = False
                    break
                if (
                    w == v
                    and vertex_map[u] != -1
                    and (vertex_map[u], candidate) not in target_adj
                ):
                    ok = False
                    break
            if ok and backtrack(pos + 1):
                return True
            vertex_map[v] = -1
        return False

    found = backtrack(0)
    return RetractionCheckResult(is_retraction=found)


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
        return FixedLengthCycleResult(
            graph=graph,
            decision="EXISTS",
            length=k,
            cycle=tuple(graph.vertices[i] for i in found),
        )
    return FixedLengthCycleResult(
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
    # Every admitted request declares this bound; the sentinel keeps direct
    # kernel callers (result replay) on the same charged accounting.
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
    from jacobian.math.graphs.morphisms._models import (
        _MAX_SEARCH_PATHS_PER_PASS,
    )

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
        return SubgraphPatternFindResult(
            pattern=request.pattern,
            host=request.host,
            decision="BUDGET_EXCEEDED",
            vertex_map=(),
        )
    if found is not None:
        return SubgraphPatternFindResult(
            pattern=request.pattern,
            host=request.host,
            decision="EXISTS",
            vertex_map=tuple(request.host.vertices[i] for i in found),
        )
    return SubgraphPatternFindResult(
        pattern=request.pattern,
        host=request.host,
        decision="DOES_NOT_EXIST",
        vertex_map=(),
    )
