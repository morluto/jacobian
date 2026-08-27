"""Exact bounded finite hypergraph operations."""

from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.hypergraphs._models import (
    CliqueExpansionRequest,
    CliqueExpansionResult,
    DualRequest,
    DualResult,
    EdgeIntersectionEntry,
    EdgeIntersectionsRequest,
    EdgeIntersectionsResult,
    FiniteHypergraph,
    HypergraphIndependenceRequest,
    HypergraphIndependenceResult,
    IncidenceGraphRequest,
    IncidenceGraphResult,
    ParametersRequest,
    ParametersResult,
    VertexDegreesRequest,
    VertexDegreesResult,
    _admit_edge_intersection_profile,
)


def compute_independence_number(
    request: HypergraphIndependenceRequest,
) -> HypergraphIndependenceResult:
    """Return an exact optimum or source-bound incumbent and sound bounds."""

    from jacobian.math.hypergraphs import _independence_z3

    return _independence_z3.solve_independence_number(request)


def verify_independence_result(result: HypergraphIndependenceResult) -> bool:
    """Verify a separately supplied independence outcome when it is bounded."""

    from jacobian.math.hypergraphs import _independence_z3

    return _independence_z3.verify_independence_result(result)


def _canonical_edges(
    hypergraph: FiniteHypergraph,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the edges with member labels in sorted canonical order."""

    return tuple(
        (edge_id, tuple(sorted(members))) for edge_id, members in hypergraph.edges
    )


def _parameters_data(
    hypergraph: FiniteHypergraph,
) -> tuple[int, int, int, int, int | None, int]:
    """Compute ``(vertex_count, edge_count, rank, corank, uniform_size, total)``."""

    edges = _canonical_edges(hypergraph)
    vertex_count = len(hypergraph.vertices)
    edge_count = len(edges)
    if edge_count == 0:
        rank = 0
        corank = 0
        uniform_size: int | None = None
        total = 0
    else:
        sizes = [len(members) for _, members in edges]
        rank = max(sizes)
        corank = min(sizes)
        total = sum(sizes)
        uniform_size = sizes[0] if all(size == sizes[0] for size in sizes) else None
    return vertex_count, edge_count, rank, corank, uniform_size, total


def _vertex_degrees_data(
    hypergraph: FiniteHypergraph,
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[int, int], ...]]:
    """Compute the vertex-degree map and degree histogram.

    ``degrees`` is a tuple of ``(vertex_label, degree)`` pairs in declared
    vertex order.  ``histogram`` is a tuple of ``(degree, count)`` pairs
    sorted by degree ascending.
    """

    degrees: dict[str, int] = dict.fromkeys(hypergraph.vertices, 0)
    for _, members in _canonical_edges(hypergraph):
        for member in members:
            degrees[member] += 1
    degree_map = tuple((vertex, degrees[vertex]) for vertex in hypergraph.vertices)
    histogram_map: dict[int, int] = {}
    for count in degrees.values():
        histogram_map[count] = histogram_map.get(count, 0) + 1
    histogram = tuple(sorted(histogram_map.items()))
    return degree_map, histogram


def _edge_intersections_data(
    hypergraph: FiniteHypergraph,
) -> tuple[
    tuple[EdgeIntersectionEntry, ...],
    tuple[tuple[int, int], ...],
    int,
    int,
    bool,
    EdgeIntersectionEntry | None,
]:
    """Compute the complete canonical indexed edge-pair intersection ledger."""

    edges = _canonical_edges(hypergraph)
    member_sets = tuple(frozenset(members) for _, members in edges)
    entries: list[EdgeIntersectionEntry] = []
    histogram_counts: dict[int, int] = {}
    maximum_intersection_size = 0
    first_linearity_violation: EdgeIntersectionEntry | None = None

    for left in range(len(edges)):
        for right in range(left + 1, len(edges)):
            intersection = tuple(sorted(member_sets[left] & member_sets[right]))
            intersection_size = len(intersection)
            entry = EdgeIntersectionEntry(
                left_edge_id=edges[left][0],
                right_edge_id=edges[right][0],
                intersection=intersection,
                intersection_size=intersection_size,
            )
            entries.append(entry)
            histogram_counts[intersection_size] = (
                histogram_counts.get(intersection_size, 0) + 1
            )
            maximum_intersection_size = max(
                maximum_intersection_size, intersection_size
            )
            if first_linearity_violation is None and intersection_size > 1:
                first_linearity_violation = entry

    pair_intersections = tuple(entries)
    histogram = tuple(sorted(histogram_counts.items()))
    return (
        pair_intersections,
        histogram,
        len(pair_intersections),
        maximum_intersection_size,
        first_linearity_violation is None,
        first_linearity_violation,
    )


def _dual_data(hypergraph: FiniteHypergraph) -> FiniteHypergraph:
    """Compute the dual hypergraph.

    The dual transposes vertices and edges: the original edge ids become the
    dual vertices, and each original vertex becomes a dual edge containing
    the original edges it belongs to.
    """

    dual_vertices = tuple(edge_id for edge_id, _ in _canonical_edges(hypergraph))
    membership: dict[str, list[str]] = {vertex: [] for vertex in hypergraph.vertices}
    for edge_id, members in _canonical_edges(hypergraph):
        for member in members:
            membership[member].append(edge_id)
    dual_edges = tuple(
        (vertex, tuple(sorted(membership[vertex]))) for vertex in hypergraph.vertices
    )
    return FiniteHypergraph(vertices=dual_vertices, edges=dual_edges)


def _incidence_graph_data(
    hypergraph: FiniteHypergraph,
) -> tuple[
    tuple[tuple[str, tuple[str, ...]], ...],
    tuple[tuple[str, tuple[str, ...]], ...],
    tuple[tuple[str, str], ...],
]:
    """Compute the bipartite incidence graph (Levi graph).

    ``vertex_incidence`` maps each vertex to the edge ids containing it in
    declared edge order.  ``edge_incidence`` maps each edge id to the
    vertices it contains in declared vertex order.  ``edges`` is the list of
    ``(vertex, edge_id)`` incidence pairs sorted by vertex then edge id.
    """

    edges = _canonical_edges(hypergraph)
    vertex_incidence: dict[str, list[str]] = {
        vertex: [] for vertex in hypergraph.vertices
    }
    for edge_id, members in edges:
        for member in members:
            vertex_incidence[member].append(edge_id)
    vertex_incidence_pairs = tuple(
        (vertex, tuple(vertex_incidence[vertex])) for vertex in hypergraph.vertices
    )
    edge_incidence_pairs = tuple((edge_id, members) for edge_id, members in edges)
    incidence_edges = tuple(
        (vertex, edge_id)
        for vertex, members in vertex_incidence_pairs
        for edge_id in members
    )
    return vertex_incidence_pairs, edge_incidence_pairs, incidence_edges


def _clique_expansion_graph(hypergraph: FiniteHypergraph) -> SimpleUndirectedGraph:
    """Compute the 2-section (primal/clique expansion) canonical graph.

    Two distinct vertices are adjacent if and only if they share at least
    one hyperedge.  Vertex labels carry over unchanged, in declared order;
    each undirected adjacency pair is emitted in lexical order, following
    the ``SimpleUndirectedGraph`` convention independently of the source
    hypergraph's declared vertex ordering.
    """

    adjacent: dict[str, set[str]] = {vertex: set() for vertex in hypergraph.vertices}
    for _, members in _canonical_edges(hypergraph):
        for i, u in enumerate(members):
            for v in members[i + 1 :]:
                adjacent[u].add(v)
                adjacent[v].add(u)
    graph_edges = tuple(
        sorted(
            (u, v) for u, neighbours in adjacent.items() for v in neighbours if u < v
        )
    )
    return SimpleUndirectedGraph(vertices=hypergraph.vertices, edges=graph_edges)


def compute_parameters(request: ParametersRequest) -> ParametersResult:
    """Compute the basic parameters of a finite hypergraph."""

    (
        vertex_count,
        edge_count,
        rank,
        corank,
        uniform_size,
        total_incidences,
    ) = _parameters_data(request.hypergraph)
    return ParametersResult(
        hypergraph=request.hypergraph,
        vertex_count=vertex_count,
        edge_count=edge_count,
        rank=rank,
        corank=corank,
        uniform_size=uniform_size,
        total_incidences=total_incidences,
    )


def verify_parameters_result(result: ParametersResult) -> bool:
    """Verify an independently supplied basic-parameter profile."""

    return (
        result.vertex_count,
        result.edge_count,
        result.rank,
        result.corank,
        result.uniform_size,
        result.total_incidences,
    ) == _parameters_data(result.hypergraph)


def compute_vertex_degrees(request: VertexDegreesRequest) -> VertexDegreesResult:
    """Compute the vertex-degree map of a finite hypergraph."""

    degrees, histogram = _vertex_degrees_data(request.hypergraph)
    return VertexDegreesResult(
        hypergraph=request.hypergraph,
        degrees=degrees,
        histogram=histogram,
    )


def verify_vertex_degrees_result(result: VertexDegreesResult) -> bool:
    """Verify an independently supplied vertex-degree profile."""

    return (result.degrees, result.histogram) == _vertex_degrees_data(result.hypergraph)


def edge_intersections(
    hypergraph: FiniteHypergraph,
) -> EdgeIntersectionsResult:
    """Return every indexed edge-pair intersection and the linearity profile."""

    _admit_edge_intersection_profile(hypergraph)
    (
        pair_intersections,
        histogram,
        pair_count,
        maximum_intersection_size,
        is_linear,
        first_linearity_violation,
    ) = _edge_intersections_data(hypergraph)
    return EdgeIntersectionsResult(
        hypergraph=hypergraph,
        pair_intersections=pair_intersections,
        pair_count=pair_count,
        histogram=histogram,
        maximum_intersection_size=maximum_intersection_size,
        is_linear=is_linear,
        first_linearity_violation=first_linearity_violation,
    )


def compute_edge_intersections(
    request: EdgeIntersectionsRequest,
) -> EdgeIntersectionsResult:
    """Compute the complete indexed edge-intersection profile."""

    return edge_intersections(request.hypergraph)


def verify_edge_intersections_result(result: EdgeIntersectionsResult) -> bool:
    """Verify an independently supplied complete edge-intersection profile."""

    _admit_edge_intersection_profile(result.hypergraph)
    return (
        result.pair_intersections,
        result.histogram,
        result.pair_count,
        result.maximum_intersection_size,
        result.is_linear,
        result.first_linearity_violation,
    ) == _edge_intersections_data(result.hypergraph)


def compute_dual(request: DualRequest) -> DualResult:
    """Compute the dual of a finite hypergraph."""

    dual = _dual_data(request.hypergraph)
    return DualResult(hypergraph=request.hypergraph, dual=dual)


def verify_dual_result(result: DualResult) -> bool:
    """Verify an independently supplied dual hypergraph."""

    return result.dual == _dual_data(result.hypergraph)


def compute_incidence_graph(
    request: IncidenceGraphRequest,
) -> IncidenceGraphResult:
    """Compute the bipartite incidence graph (Levi graph) of a hypergraph."""

    vertex_incidence, edge_incidence, edges = _incidence_graph_data(request.hypergraph)
    return IncidenceGraphResult(
        hypergraph=request.hypergraph,
        vertex_incidence=vertex_incidence,
        edge_incidence=edge_incidence,
        edges=edges,
    )


def verify_incidence_graph_result(result: IncidenceGraphResult) -> bool:
    """Verify an independently supplied incidence-graph profile."""

    return (
        result.vertex_incidence,
        result.edge_incidence,
        result.edges,
    ) == _incidence_graph_data(result.hypergraph)


def compute_clique_expansion(
    request: CliqueExpansionRequest,
) -> CliqueExpansionResult:
    """Compute the 2-section (primal/clique expansion) of a hypergraph."""

    return CliqueExpansionResult(
        hypergraph=request.hypergraph,
        graph=_clique_expansion_graph(request.hypergraph),
    )


def verify_clique_expansion_result(result: CliqueExpansionResult) -> bool:
    """Verify an independently supplied clique-expansion graph."""

    return result.graph == _clique_expansion_graph(result.hypergraph)
