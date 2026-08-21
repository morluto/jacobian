"""Domain-owned graph coloring and independent set operations."""

from __future__ import annotations

from jacobian.math.graphs.coloring._models import (
    EdgeColoringCheckRequest,
    EdgeColoringCheckResult,
    EdgeKColorabilityRequest,
    EdgeKColorabilityResult,
    GraphEdgeList,
    KColorabilityRequest,
    KColorabilityResult,
    MaximalIndependentSetRequest,
    MaximalIndependentSetResult,
)


def compute_k_colorability(request: KColorabilityRequest) -> KColorabilityResult:
    import z3  # type: ignore[import-untyped]

    solver = z3.Solver()
    colors = [z3.Int(f"color_{vertex}") for vertex in range(request.graph.vertex_count)]
    solver.add(*(z3.And(color >= 0, color < request.colors) for color in colors))
    solver.add(*(colors[u] != colors[v] for u, v in request.graph.edges))
    if solver.check() == z3.sat:
        model = solver.model()
        coloring = tuple(model.eval(color).as_long() for color in colors)
        return KColorabilityResult(
            colorable=True,
            coloring=coloring,
            vertex_count=request.graph.vertex_count,
            colors=request.colors,
        )
    return KColorabilityResult(
        colorable=False,
        vertex_count=request.graph.vertex_count,
        colors=request.colors,
    )


def compute_maximal_independent_set_decision(
    request: MaximalIndependentSetRequest,
) -> MaximalIndependentSetResult:
    """Decide maximal independence and return the first canonical obstruction."""
    candidate = frozenset(request.candidate_set)
    edges = tuple(sorted((min(u, v), max(u, v)) for u, v in request.graph.edges))
    for edge in edges:
        if edge[0] in candidate and edge[1] in candidate:
            return MaximalIndependentSetResult(
                decision="NOT_INDEPENDENT",
                blocking_edge=edge,
            )

    adjacency: list[set[int]] = [set() for _ in range(request.graph.vertex_count)]
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    for vertex in range(request.graph.vertex_count):
        if vertex not in candidate and adjacency[vertex].isdisjoint(candidate):
            return MaximalIndependentSetResult(
                decision="INDEPENDENT_NOT_MAXIMAL",
                addable_vertex=vertex,
            )
    return MaximalIndependentSetResult(decision="MAXIMAL")


def _incident_edge_index_pairs(graph: GraphEdgeList) -> list[tuple[int, int]]:
    """Return pairs of edge indices that share a vertex (must differ in color)."""
    incidence: dict[int, list[int]] = {}
    for edge_index, (u, v) in enumerate(graph.edges):
        incidence.setdefault(u, []).append(edge_index)
        incidence.setdefault(v, []).append(edge_index)
    pairs: list[tuple[int, int]] = []
    for indices in incidence.values():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                pairs.append((indices[a], indices[b]))
    return pairs


def compute_edge_k_colorability(request: EdgeKColorabilityRequest) -> EdgeKColorabilityResult:
    """Decide whether a simple graph admits a proper ``k``-edge-coloring.

    A proper edge coloring assigns each edge a color in ``0..k-1`` such that
    incident edges receive distinct colors.  Uses a bounded Z3 SAT search and
    returns one coloring witness when one exists.
    """
    import z3

    edges = request.graph.edges
    if not edges:
        return EdgeKColorabilityResult(
            colorable=True,
            coloring=(),
            edge_count=0,
            colors=request.colors,
        )
    solver = z3.Solver()
    edge_colors = [z3.Int(f"c_{i}") for i in range(len(edges))]
    solver.add(*(z3.And(c >= 0, c < request.colors) for c in edge_colors))
    for a, b in _incident_edge_index_pairs(request.graph):
        solver.add(edge_colors[a] != edge_colors[b])
    if solver.check() == z3.sat:
        model = solver.model()
        coloring = tuple(model.eval(c).as_long() for c in edge_colors)
        return EdgeKColorabilityResult(
            colorable=True,
            coloring=coloring,
            edge_count=len(edges),
            colors=request.colors,
        )
    return EdgeKColorabilityResult(
        colorable=False,
        edge_count=len(edges),
        colors=request.colors,
    )


def compute_edge_coloring_check(request: EdgeColoringCheckRequest) -> EdgeColoringCheckResult:
    """Validate a submitted edge-to-color assignment as a proper edge coloring."""
    edges = request.graph.edges
    coloring = request.coloring
    for a, b in _incident_edge_index_pairs(request.graph):
        if coloring[a] == coloring[b]:
            u_a, v_a = edges[a]
            return EdgeColoringCheckResult(
                proper=False,
                blocking_edge=(min(u_a, v_a), max(u_a, v_a)),
            )
    return EdgeColoringCheckResult(proper=True, blocking_edge=None)
