"""Exact bounded finite hypergraph operations."""

import unicodedata

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_HYPERGRAPH_INDEPENDENCE_INCIDENCES,
    MAX_HYPERGRAPH_INDEPENDENCE_VERTICES,
    MAX_INDUCED_PROFILE_RESULT_BYTES,
    MAX_MATCHING_EDGES,
    MAX_MATCHING_RESULT_BYTES,
    MAX_TRANSVERSAL_RESULT_BYTES,
    MAX_TRANSVERSAL_SEARCH_WORK,
    MAX_VERTICES,
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
    InducedTypeProfileEntry,
    InducedTypeProfileRequest,
    InducedTypeProfileResult,
    MaximumEdgeMatchingRequest,
    MaximumEdgeMatchingResult,
    MinimumTransversalRequest,
    MinimumTransversalResult,
    ParametersRequest,
    ParametersResult,
    VertexDegreesRequest,
    VertexDegreesResult,
    _admit_edge_intersection_profile,
    _induced_type_profile_admission_plan,
    _InducedTypeProfileAdmissionPlan,
    _maximum_edge_matching_result_bytes,
    _minimum_transversal_result_bytes,
    _minimum_transversal_search_plan,
    _validation_error,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _admit_independence(request: HypergraphIndependenceRequest) -> None:
    if any(not members for _, members in request.hypergraph.edges):
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.independence_number.empty_edge",
            message="independence-number search does not admit empty edges",
        )
    total_incidences = sum(len(members) for _, members in request.hypergraph.edges)
    if len(request.hypergraph.vertices) > MAX_HYPERGRAPH_INDEPENDENCE_VERTICES:
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.independence_number.vertex_bound",
            message=(
                "independence-number search exceeds the "
                f"{MAX_HYPERGRAPH_INDEPENDENCE_VERTICES}-vertex solver bound"
            ),
        )
    if total_incidences > MAX_HYPERGRAPH_INDEPENDENCE_INCIDENCES:
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.independence_number.incidence_bound",
            message=(
                "independence-number search exceeds the "
                f"{MAX_HYPERGRAPH_INDEPENDENCE_INCIDENCES}-incidence solver bound"
            ),
        )


def _admit_dual(request: DualRequest) -> None:
    if len(request.hypergraph.edges) > MAX_VERTICES:
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.dual.vertex_bound",
            message=(
                "hypergraph dual exceeds the "
                f"{MAX_VERTICES}-vertex representation bound"
            ),
        )


def _admit_clique_expansion(request: CliqueExpansionRequest) -> None:
    if any(
        not unicodedata.is_normalized("NFC", vertex)
        for vertex in request.hypergraph.vertices
    ):
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.clique_expansion.nfc_vertices",
            message="clique expansion requires NFC-normalized vertex labels",
        )


def _admit_maximum_edge_matching(request: MaximumEdgeMatchingRequest) -> None:
    edge_ids = tuple(edge_id for edge_id, _ in request.hypergraph.edges)
    nonempty_edge_count = sum(bool(members) for _, members in request.hypergraph.edges)
    if nonempty_edge_count > MAX_MATCHING_EDGES:
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.maximum_edge_matching.search_bound",
            message=(
                "maximum edge matching search exceeds the "
                f"{MAX_MATCHING_EDGES}-edge exact search bound"
            ),
        )
    if (
        _maximum_edge_matching_result_bytes(request.hypergraph, edge_ids)
        > MAX_MATCHING_RESULT_BYTES
    ):
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.maximum_edge_matching.result_bound",
            message=(
                "the maximum edge matching result retains its source hypergraph "
                f"and would exceed the {MAX_MATCHING_RESULT_BYTES}-byte "
                "canonical output limit; shorten labels or reduce the edge family"
            ),
        )


def compute_independence_number(
    request: HypergraphIndependenceRequest,
) -> HypergraphIndependenceResult:
    """Return an exact optimum or source-bound incumbent and sound bounds."""

    from jacobian.math.combinatorics.finite_structures.hypergraphs import (
        _independence_z3,
    )

    _admit_independence(request)
    return _independence_z3.solve_independence_number(request)


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


def compute_vertex_degrees(request: VertexDegreesRequest) -> VertexDegreesResult:
    """Compute the vertex-degree map of a finite hypergraph."""

    degrees, histogram = _vertex_degrees_data(request.hypergraph)
    return VertexDegreesResult(
        hypergraph=request.hypergraph,
        degrees=degrees,
        histogram=histogram,
    )


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


def compute_dual(request: DualRequest) -> DualResult:
    """Compute the dual of a finite hypergraph."""

    _admit_dual(request)
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

    _admit_clique_expansion(request)
    return CliqueExpansionResult(
        hypergraph=request.hypergraph,
        graph=_clique_expansion_graph(request.hypergraph),
    )


def _induced_type_profile_data(
    plan: _InducedTypeProfileAdmissionPlan,
) -> tuple[tuple[tuple[str, ...], int], ...]:
    """Compute one ``(vertex_subset, induced_edge_count)`` pair per k-subset.

    For each k-subset ``S`` of the declared vertices, the induced edge count
    is the number of distinct nonempty edges ``e ∩ S`` arising from the source
    hypergraph's edges.  Subsets are emitted in lexicographic vertex order.
    """
    rows: list[tuple[tuple[str, ...], int]] = []
    for subset in plan.expected_subsets:
        subset_set = frozenset(subset)
        distinct_edges: set[frozenset[str]] = set()
        for members in plan.edge_sets:
            induced = members & subset_set
            if induced:
                distinct_edges.add(induced)
        rows.append((subset, len(distinct_edges)))
    return tuple(rows)


def compute_induced_type_profile(
    request: InducedTypeProfileRequest,
) -> InducedTypeProfileResult:
    """Compute the induced uniform type profile of a finite hypergraph."""

    plan = _induced_type_profile_admission_plan(
        request.hypergraph,
        request.subset_size,
    )
    if plan.result_bytes > MAX_INDUCED_PROFILE_RESULT_BYTES:
        raise _validation_error(
            "the induced type profile would exceed the "
            f"{MAX_INDUCED_PROFILE_RESULT_BYTES}-byte canonical output limit; "
            "shorten vertex labels or reduce the profile"
        )
    rows = _induced_type_profile_data(plan)
    entries = tuple(
        InducedTypeProfileEntry(vertex_subset=subset, induced_edge_count=count)
        for subset, count in rows
    )
    return InducedTypeProfileResult._from_kernel(
        hypergraph=request.hypergraph,
        subset_size=request.subset_size,
        entries=entries,
    )


def _minimum_transversal_data(
    plan: tuple[tuple[str, ...], tuple[frozenset[str], ...], int, int],
) -> tuple[tuple[str, ...], int]:
    """Return one minimum transversal (declared vertex order) and its size.

    The empty hyperedge family admits the empty transversal.  Otherwise the
    search enumerates vertex subsets by increasing cardinality; the first
    hitting set found is minimum by construction.
    """
    from itertools import combinations

    vertices, edge_sets, search_depth, _search_work = plan
    if not edge_sets:
        return (), 0
    for size in range(1, search_depth + 1):
        for combo in combinations(vertices, size):
            candidate = frozenset(combo)
            if all(candidate & edge for edge in edge_sets):
                ordered = tuple(vertex for vertex in vertices if vertex in candidate)
                return ordered, size
    # Unreachable: the full vertex set hits every nonempty edge.
    raise AssertionError("minimum transversal search exhausted all vertices")


def _admit_minimum_transversal(
    hypergraph: FiniteHypergraph,
) -> tuple[tuple[str, ...], tuple[frozenset[str], ...], int, int]:
    """Admit a transversal request and return its reusable search plan."""

    if any(not members for _, members in hypergraph.edges):
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.minimum_transversal.empty_edge",
            message="minimum transversal search does not admit empty edges",
        )
    plan = _minimum_transversal_search_plan(hypergraph)
    active_vertices, _, _, search_work = plan
    if search_work > MAX_TRANSVERSAL_SEARCH_WORK:
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.minimum_transversal.search_bound",
            message=(
                "minimum transversal search exceeds the "
                f"{MAX_TRANSVERSAL_SEARCH_WORK}-check exact search bound"
            ),
        )
    if (
        _minimum_transversal_result_bytes(hypergraph, active_vertices)
        > MAX_TRANSVERSAL_RESULT_BYTES
    ):
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.minimum_transversal.result_bound",
            message=(
                "the minimum transversal result retains its source hypergraph "
                f"and would exceed the {MAX_TRANSVERSAL_RESULT_BYTES}-byte "
                "canonical output limit; shorten labels or reduce the edge family"
            ),
        )
    return plan


def compute_minimum_transversal(
    request: MinimumTransversalRequest,
) -> MinimumTransversalResult:
    """Compute an exact minimum-cardinality transversal of a finite hypergraph."""

    plan = _admit_minimum_transversal(request.hypergraph)
    transversal, cardinality = _minimum_transversal_data(plan)
    return MinimumTransversalResult(
        hypergraph=request.hypergraph,
        transversal=transversal,
        cardinality=cardinality,
    )


def _maximum_edge_matching_data(
    hypergraph: FiniteHypergraph,
) -> tuple[tuple[str, ...], int]:
    """Return one maximum matching (declared edge order) and its size.

    The empty edge family admits the empty matching.  Otherwise the search
    enumerates edge subsets by decreasing cardinality; the first pairwise-
    disjoint family found is maximum by construction.
    """
    from itertools import combinations

    edges = _canonical_edges(hypergraph)
    edge_ids = tuple(edge_id for edge_id, _ in edges)
    empty_edge_ids = tuple(edge_id for edge_id, members in edges if not members)
    search_edges = tuple((edge_id, members) for edge_id, members in edges if members)
    search_edge_ids = tuple(edge_id for edge_id, _ in search_edges)
    edge_sets = tuple(frozenset(members) for _, members in search_edges)
    if not search_edges:
        return edge_ids, len(edge_ids)
    for size in range(len(search_edges), 0, -1):
        for combo in combinations(range(len(search_edges)), size):
            picked = [edge_sets[i] for i in combo]
            disjoint = True
            for i in range(len(picked)):
                for j in range(i + 1, len(picked)):
                    if picked[i] & picked[j]:
                        disjoint = False
                        break
                if not disjoint:
                    break
            if disjoint:
                selected_ids = set(empty_edge_ids)
                selected_ids.update(search_edge_ids[i] for i in combo)
                ordered = tuple(
                    edge_id for edge_id in edge_ids if edge_id in selected_ids
                )
                return ordered, len(empty_edge_ids) + size
    return (), 0


def compute_maximum_edge_matching(
    request: MaximumEdgeMatchingRequest,
) -> MaximumEdgeMatchingResult:
    """Compute an exact maximum-cardinality edge matching of a finite hypergraph."""

    _admit_maximum_edge_matching(request)
    matching, count = _maximum_edge_matching_data(request.hypergraph)
    return MaximumEdgeMatchingResult(
        hypergraph=request.hypergraph,
        matching=matching,
        count=count,
    )
