"""Exact bounded finite hypergraph operations."""

from jacobian.math.hypergraphs._models import (
    CliqueExpansionRequest,
    CliqueExpansionResult,
    DualRequest,
    DualResult,
    FiniteHypergraph,
    IncidenceGraphRequest,
    IncidenceGraphResult,
    ParametersRequest,
    ParametersResult,
    VertexDegreesRequest,
    VertexDegreesResult,
)


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


def _clique_expansion_data(
    hypergraph: FiniteHypergraph,
) -> tuple[
    tuple[str, ...],
    tuple[tuple[str, tuple[str, ...]], ...],
    tuple[tuple[str, str], ...],
]:
    """Compute the 2-section (primal/clique expansion) graph.

    Two distinct vertices are adjacent if they share at least one hyperedge.
    ``adjacency`` maps each vertex to its neighbours in declared vertex order.
    ``edges`` lists each adjacency pair ``(u, v)`` with ``u`` before ``v`` in
    declared order, sorted by the first component then the second.
    """

    edges = _canonical_edges(hypergraph)
    index = {vertex: i for i, vertex in enumerate(hypergraph.vertices)}
    adjacent: list[set[str]] = [set() for _ in hypergraph.vertices]
    for _, members in edges:
        n = len(members)
        for i in range(n):
            for j in range(i + 1, n):
                u, v = members[i], members[j]
                if u == v:
                    continue
                adjacent[index[u]].add(v)
                adjacent[index[v]].add(u)
    vertices = hypergraph.vertices
    adjacency = tuple(
        (
            vertex,
            tuple(
                neighbour
                for neighbour in hypergraph.vertices
                if neighbour in adjacent[index[vertex]]
            ),
        )
        for vertex in hypergraph.vertices
    )
    edge_list: list[tuple[str, str]] = []
    for i, vertex in enumerate(hypergraph.vertices):
        for neighbour in adjacent[i]:
            j = index[neighbour]
            if j > i:
                edge_list.append((vertex, neighbour))
    edge_list.sort(key=lambda pair: (index[pair[0]], index[pair[1]]))
    return vertices, adjacency, tuple(edge_list)


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


def compute_vertex_degrees(request: VertexDegreesRequest) -> VertexDegreesResult:
    """Compute the vertex-degree map of a finite hypergraph."""

    degrees, histogram = _vertex_degrees_data(request.hypergraph)
    return VertexDegreesResult(
        hypergraph=request.hypergraph,
        degrees=degrees,
        histogram=histogram,
    )


def compute_dual(request: DualRequest) -> DualResult:
    """Compute the dual of a finite hypergraph."""

    dual = _dual_data(request.hypergraph)
    return DualResult(hypergraph=request.hypergraph, dual=dual)


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


def compute_clique_expansion(
    request: CliqueExpansionRequest,
) -> CliqueExpansionResult:
    """Compute the 2-section (primal/clique expansion) of a hypergraph."""

    vertices, adjacency, edges = _clique_expansion_data(request.hypergraph)
    return CliqueExpansionResult(
        hypergraph=request.hypergraph,
        vertices=vertices,
        adjacency=adjacency,
        edges=edges,
    )
