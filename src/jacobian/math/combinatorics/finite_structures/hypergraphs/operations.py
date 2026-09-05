"""Exact bounded finite hypergraph operations."""

import unicodedata
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_HYPERGRAPH_INDEPENDENCE_INCIDENCES,
    MAX_HYPERGRAPH_INDEPENDENCE_VERTICES,
    MAX_MATCHING_EDGES,
    MAX_MATCHING_SEARCH_WORK,
    MAX_TRANSVERSAL_SEARCH_WORK,
    MAX_VERTICES,
    MAX_WEIGHTED_PACKING_SEARCH_WORK,
    CliqueExpansionResult,
    DualResult,
    EdgeIntersectionEntry,
    EdgeIntersectionGraphResult,
    EdgeIntersectionsResult,
    EdgeWeight,
    FiniteHypergraph,
    HypergraphIndependenceBudget,
    HypergraphIndependenceResult,
    IncidenceGraphResult,
    InducedTypeProfileEntry,
    InducedTypeProfileResult,
    MaximumEdgeMatchingResult,
    MinimumTransversalResult,
    ParametersResult,
    VertexDegreesResult,
    WeightedPackingResult,
    _admit_edge_intersection_profile,
    _induced_type_profile_admission_plan,
    _InducedTypeProfileAdmissionPlan,
    _minimum_transversal_search_plan,
    _TransversalSearchPlan,
)
from jacobian.math.graphs.values import (
    MAX_GRAPH_LABEL_BYTES,
    MAX_INDEXED_SIMPLE_GRAPH_VERTICES,
    SimpleUndirectedGraph,
)

__all__ = [
    "clique_expansion",
    "dual",
    "edge_intersection_graph",
    "edge_intersections",
    "incidence_graph",
    "independence_number",
    "induced_type_profile",
    "maximum_edge_matching",
    "maximum_weight_packing",
    "minimum_transversal",
    "parameters",
    "vertex_degrees",
]


def _admit_independence(hypergraph: FiniteHypergraph) -> tuple[str, ...] | None:
    if any(not members for _, members in hypergraph.edges):
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.independence_number.empty_edge",
            message="independence-number search does not admit empty edges",
        )
    total_incidences = sum(len(members) for _, members in hypergraph.edges)
    # The canonical carrier bounds this linear scan by 256 vertices,
    # 12,000 edges and 36,000 incidences. Singleton constraints force these
    # exclusions; if they hit every edge, all remaining vertices attain the
    # resulting upper bound. No greedy scan or solver encoding is needed.
    forbidden = {members[0] for _, members in hypergraph.edges if len(members) == 1}
    if all(
        any(vertex in forbidden for vertex in members)
        for _, members in hypergraph.edges
    ):
        return tuple(
            vertex for vertex in hypergraph.vertices if vertex not in forbidden
        )
    if len(hypergraph.vertices) > MAX_HYPERGRAPH_INDEPENDENCE_VERTICES:
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
    return None


def _admit_dual(hypergraph: FiniteHypergraph) -> None:
    if len(hypergraph.edges) > MAX_VERTICES:
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.dual.vertex_bound",
            message=(
                "hypergraph dual exceeds the "
                f"{MAX_VERTICES}-vertex representation bound"
            ),
        )


def _admit_clique_expansion(hypergraph: FiniteHypergraph) -> None:
    if any(
        not unicodedata.is_normalized("NFC", vertex) for vertex in hypergraph.vertices
    ):
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.clique_expansion.nfc_vertices",
            message="clique expansion requires NFC-normalized vertex labels",
        )


def _admit_edge_intersection_graph(
    hypergraph: FiniteHypergraph,
) -> tuple[tuple[str, str], ...]:
    if any(
        not unicodedata.is_normalized("NFC", edge_id) for edge_id, _ in hypergraph.edges
    ):
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.edge_intersection_graph.nfc_edge_ids",
            message=("edge-intersection graph requires NFC-normalized edge IDs"),
        )
    # Reject empty edge IDs: they would produce empty graph vertex labels
    # that are incompatible with downstream consumers (e.g. GraphVertexLabel
    # enforces min_length=1).
    for edge_id, _ in hypergraph.edges:
        if not edge_id:
            raise OperationDomainValidationError(
                location=("hypergraph",),
                code="hypergraph.edge_intersection_graph.nonempty_edge_ids",
                message="edge-intersection graph edge IDs must be nonempty",
            )
    # The edge-intersection graph maps each hyperedge to a graph vertex, so
    # the number of hyperedges must fit the SimpleUndirectedGraph carrier.
    if len(hypergraph.edges) > MAX_INDEXED_SIMPLE_GRAPH_VERTICES:
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.edge_intersection_graph.carrier_vertex_bound",
            message=(
                "edge-intersection graph exceeds the "
                f"{MAX_INDEXED_SIMPLE_GRAPH_VERTICES}-vertex graph carrier bound"
            ),
        )
    # Vertex labels of the target graph are the edge IDs of the source
    # hypergraph; they must fit the graph label length to compose with
    # downstream graph operations.
    for edge_id, _ in hypergraph.edges:
        if len(edge_id) == 0 or len(edge_id.encode("utf-8")) > MAX_GRAPH_LABEL_BYTES:
            raise OperationDomainValidationError(
                location=("hypergraph",),
                code="hypergraph.edge_intersection_graph.label_length",
                message=(
                    "edge-intersection graph edge IDs must be nonempty "
                    f"and at most {MAX_GRAPH_LABEL_BYTES} UTF-8 bytes "
                    "to fit the graph carrier"
                ),
            )
    # Build the exact graph edge set once for admission and reuse it for the
    # result.  Charging every admitted input as a complete graph rejects sparse
    # graphs that are well within the retained edge-cardinality boundary.
    edge_ids = tuple(edge_id for edge_id, _ in hypergraph.edges)
    member_sets = tuple(frozenset(members) for _, members in hypergraph.edges)
    graph_edges = tuple(
        (edge_ids[left], edge_ids[right])
        if edge_ids[left] <= edge_ids[right]
        else (edge_ids[right], edge_ids[left])
        for left in range(len(edge_ids))
        for right in range(left + 1, len(edge_ids))
        if member_sets[left] & member_sets[right]
    )
    return graph_edges


def _conflict_components(
    edge_sets: tuple[frozenset[str], ...],
) -> tuple[tuple[int, ...], ...]:
    """Partition candidate positions into conflict-connected components.

    Two candidates conflict when their member sets intersect. Union-find over
    the vertex incidence index builds the components in near-linear
    incidence work without enumerating conflict pairs: for each vertex, all
    candidates containing it are united. Candidates from distinct components
    are pairwise disjoint, so each component is an independent matching
    subproblem and a global optimum is the union of component optima.
    Components are returned in order of their smallest position, with
    positions ascending inside each component.
    """

    parent = list(range(len(edge_sets)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    first_seen: dict[str, int] = {}
    for index, members in enumerate(edge_sets):
        for vertex in members:
            seen = first_seen.get(vertex)
            if seen is None:
                first_seen[vertex] = index
            else:
                root_a, root_b = find(index), find(seen)
                if root_a != root_b:
                    parent[max(root_a, root_b)] = min(root_a, root_b)
    groups: dict[int, list[int]] = {}
    for index in range(len(edge_sets)):
        groups.setdefault(find(index), []).append(index)
    return tuple(
        tuple(group)
        for _, group in sorted(
            ((min(group), tuple(group)) for group in groups.values()),
            key=lambda pair: pair[0],
        )
    )


def _matching_search_plan(
    hypergraph: FiniteHypergraph,
) -> tuple[
    tuple[str, ...], tuple[frozenset[str], ...], tuple[tuple[int, ...], ...], int
]:
    """Build the reusable conflict-component search plan for one matching request."""

    edges = _canonical_edges(hypergraph)
    search_edges = tuple((edge_id, members) for edge_id, members in edges if members)
    search_edge_ids = tuple(edge_id for edge_id, _ in search_edges)
    edge_sets = tuple(frozenset(members) for _, members in search_edges)
    components = _conflict_components(edge_sets)
    search_work = sum(
        (1 << len(component)) * len(component) * len(component)
        for component in components
    )
    return search_edge_ids, edge_sets, components, search_work


def _admit_maximum_edge_matching(
    hypergraph: FiniteHypergraph,
) -> tuple[
    tuple[str, ...], tuple[frozenset[str], ...], tuple[tuple[int, ...], ...], int
]:
    plan = _matching_search_plan(hypergraph)
    _, _, components, search_work = plan
    if any(len(component) > MAX_MATCHING_EDGES for component in components):
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.maximum_edge_matching.search_bound",
            message=(
                "maximum edge matching search exceeds the "
                f"{MAX_MATCHING_EDGES}-edge exact search bound"
            ),
        )
    if search_work > MAX_MATCHING_SEARCH_WORK:
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.maximum_edge_matching.search_bound",
            message=(
                "maximum edge matching search exceeds the "
                f"{MAX_MATCHING_SEARCH_WORK:,}-check exact search bound"
            ),
        )
    return plan


def independence_number(
    hypergraph: FiniteHypergraph,
    resource_budget: HypergraphIndependenceBudget | None = None,
) -> HypergraphIndependenceResult:
    """Return an exact optimum or source-bound incumbent and sound bounds."""

    resource_budget = resource_budget or HypergraphIndependenceBudget()
    trivial_witness = _admit_independence(hypergraph)
    if trivial_witness is not None:
        return HypergraphIndependenceResult._from_kernel(
            hypergraph=hypergraph,
            resource_budget=resource_budget,
            status="EXACT",
            independence_number=len(trivial_witness),
            incumbent_vertices=trivial_witness,
            upper_bound=len(trivial_witness),
            solver_calls=0,
            wall_budget_exhausted=False,
            termination_reason="SPECIAL_CASE",
            detail="singleton-forbidden vertices hit every edge; all other vertices form a maximum independent set",
        )

    from jacobian.math.combinatorics.finite_structures.hypergraphs import (
        _independence_z3,
    )

    return _independence_z3.solve_independence_number(hypergraph, resource_budget)


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


def parameters(hypergraph: FiniteHypergraph) -> ParametersResult:
    """Compute the basic parameters of a finite hypergraph."""

    (
        vertex_count,
        edge_count,
        rank,
        corank,
        uniform_size,
        total_incidences,
    ) = _parameters_data(hypergraph)
    return ParametersResult(
        hypergraph=hypergraph,
        vertex_count=vertex_count,
        edge_count=edge_count,
        rank=rank,
        corank=corank,
        uniform_size=uniform_size,
        total_incidences=total_incidences,
    )


def vertex_degrees(hypergraph: FiniteHypergraph) -> VertexDegreesResult:
    """Compute the vertex-degree map of a finite hypergraph."""

    degrees, histogram = _vertex_degrees_data(hypergraph)
    return VertexDegreesResult(
        hypergraph=hypergraph,
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
    return EdgeIntersectionsResult._from_kernel(
        hypergraph=hypergraph,
        pair_intersections=pair_intersections,
        pair_count=pair_count,
        histogram=histogram,
        maximum_intersection_size=maximum_intersection_size,
        is_linear=is_linear,
        first_linearity_violation=first_linearity_violation,
    )


def dual(hypergraph: FiniteHypergraph) -> DualResult:
    """Compute the dual of a finite hypergraph."""

    _admit_dual(hypergraph)
    return DualResult(hypergraph=hypergraph, dual=_dual_data(hypergraph))


def incidence_graph(hypergraph: FiniteHypergraph) -> IncidenceGraphResult:
    """Compute the bipartite incidence graph (Levi graph) of a hypergraph."""

    vertex_incidence, edge_incidence, edges = _incidence_graph_data(hypergraph)
    return IncidenceGraphResult(
        hypergraph=hypergraph,
        vertex_incidence=vertex_incidence,
        edge_incidence=edge_incidence,
        edges=edges,
    )


def clique_expansion(hypergraph: FiniteHypergraph) -> CliqueExpansionResult:
    """Compute the 2-section (primal/clique expansion) of a hypergraph."""

    _admit_clique_expansion(hypergraph)
    return CliqueExpansionResult(
        hypergraph=hypergraph,
        graph=_clique_expansion_graph(hypergraph),
    )


def _edge_intersection_graph_data(
    hypergraph: FiniteHypergraph,
    graph_edges: tuple[tuple[str, str], ...],
) -> SimpleUndirectedGraph:
    """Compute the canonical edge-intersection graph.

    The graph's vertices are the hypergraph's edge IDs in declared order.
    Two vertices are adjacent if and only if the corresponding hyperedges
    have nonempty intersection.  Each undirected adjacency pair is emitted
    in lexical order, following the ``SimpleUndirectedGraph`` convention
    independently of the source hypergraph's declared edge ordering.
    """

    edge_ids = tuple(edge_id for edge_id, _ in _canonical_edges(hypergraph))
    return SimpleUndirectedGraph(vertices=edge_ids, edges=graph_edges)


def edge_intersection_graph(
    hypergraph: FiniteHypergraph,
) -> EdgeIntersectionGraphResult:
    """Compute the edge-intersection graph of a finite hypergraph."""

    graph_edges = _admit_edge_intersection_graph(hypergraph)
    return EdgeIntersectionGraphResult(
        hypergraph=hypergraph,
        graph=_edge_intersection_graph_data(hypergraph, graph_edges),
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


def induced_type_profile(
    hypergraph: FiniteHypergraph,
    subset_size: int,
) -> InducedTypeProfileResult:
    """Compute the induced uniform type profile of a finite hypergraph."""

    plan = _induced_type_profile_admission_plan(
        hypergraph,
        subset_size,
    )
    rows = _induced_type_profile_data(plan)
    entries = tuple(
        InducedTypeProfileEntry(vertex_subset=subset, induced_edge_count=count)
        for subset, count in rows
    )
    return InducedTypeProfileResult._from_kernel(
        hypergraph=hypergraph,
        subset_size=subset_size,
        entries=entries,
    )


def _minimum_component_transversal(
    vertices: tuple[str, ...],
    edge_sets: tuple[frozenset[str], ...],
    search_depth: int,
) -> tuple[str, ...]:
    """Return one minimum transversal of one residual component.

    Vertex subsets are enumerated by increasing cardinality in declared
    vertex order; the first hitting set found is minimum by construction.
    """

    from itertools import combinations

    if not edge_sets:
        return ()
    for size in range(1, search_depth + 1):
        for combo in combinations(vertices, size):
            candidate = frozenset(combo)
            if all(candidate & edge for edge in edge_sets):
                return combo
    # Unreachable: the component vertex set hits every component edge.
    raise AssertionError("minimum transversal search exhausted all vertices")


def _minimum_transversal_data(
    plan: _TransversalSearchPlan,
) -> tuple[tuple[str, ...], int]:
    """Return one minimum transversal (declared vertex order) and its size.

    Forced singleton vertices join every optimum; each residual component
    contributes its own minimum. The empty hyperedge family admits the empty
    transversal.
    """

    chosen = set(plan.forced_vertices)
    for component in plan.components:
        chosen.update(
            _minimum_component_transversal(
                component.vertices, component.edge_sets, component.search_depth
            )
        )
    ordered = tuple(vertex for vertex in plan.vertices if vertex in chosen)
    return ordered, len(ordered)


def _admit_minimum_transversal(
    hypergraph: FiniteHypergraph,
) -> _TransversalSearchPlan:
    """Admit a transversal request and return its reusable search plan."""

    if any(not members for _, members in hypergraph.edges):
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.minimum_transversal.empty_edge",
            message="minimum transversal search does not admit empty edges",
        )
    plan = _minimum_transversal_search_plan(hypergraph)
    if plan.search_work > MAX_TRANSVERSAL_SEARCH_WORK:
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.minimum_transversal.search_bound",
            message=(
                "minimum transversal search exceeds the "
                f"{MAX_TRANSVERSAL_SEARCH_WORK}-check exact search bound"
            ),
        )
    return plan


def minimum_transversal(hypergraph: FiniteHypergraph) -> MinimumTransversalResult:
    """Compute an exact minimum-cardinality transversal of a finite hypergraph."""

    plan = _admit_minimum_transversal(hypergraph)
    transversal, cardinality = _minimum_transversal_data(plan)
    return MinimumTransversalResult(
        hypergraph=hypergraph,
        transversal=transversal,
        cardinality=cardinality,
    )


def _maximum_component_matching(
    edge_sets: tuple[frozenset[str], ...],
    component: tuple[int, ...],
) -> tuple[int, ...]:
    """Return the first (combo-order) maximum disjoint subfamily of one component.

    Subsets are enumerated by decreasing cardinality over positions into
    ``edge_sets``; the first pairwise-disjoint family found is maximum by
    construction. A singleton component is always taken in full.
    """

    from itertools import combinations

    members = [edge_sets[index] for index in component]
    for size in range(len(component), 0, -1):
        for combo in combinations(range(len(component)), size):
            disjoint = True
            for left in range(len(combo)):
                for right in range(left + 1, len(combo)):
                    if members[combo[left]] & members[combo[right]]:
                        disjoint = False
                        break
                if not disjoint:
                    break
            if disjoint:
                return tuple(component[position] for position in combo)
    return ()


def _maximum_edge_matching_data(
    plan: tuple[
        tuple[str, ...], tuple[frozenset[str], ...], tuple[tuple[int, ...], ...], int
    ],
    edge_ids: tuple[str, ...],
    empty_edge_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], int]:
    """Return one maximum matching (declared edge order) and its size.

    The empty edge family admits the empty matching. Otherwise every
    conflict-component optimum is computed by bounded exhaustive search and
    the union — plus the mandatory empty-edge prefix — is maximum because
    components share no vertices.
    """

    search_edge_ids, edge_sets, components, _ = plan
    if not search_edge_ids:
        return edge_ids, len(edge_ids)
    selected: set[int] = set()
    for component in components:
        selected.update(_maximum_component_matching(edge_sets, component))
    selected_ids = set(empty_edge_ids)
    selected_ids.update(search_edge_ids[index] for index in selected)
    ordered = tuple(edge_id for edge_id in edge_ids if edge_id in selected_ids)
    return ordered, len(selected_ids)


def maximum_edge_matching(hypergraph: FiniteHypergraph) -> MaximumEdgeMatchingResult:
    """Compute an exact maximum-cardinality edge matching of a finite hypergraph."""

    plan = _admit_maximum_edge_matching(hypergraph)
    edges = _canonical_edges(hypergraph)
    edge_ids = tuple(edge_id for edge_id, _ in edges)
    empty_edge_ids = tuple(edge_id for edge_id, members in edges if not members)
    matching, count = _maximum_edge_matching_data(plan, edge_ids, empty_edge_ids)
    return MaximumEdgeMatchingResult(
        hypergraph=hypergraph,
        matching=matching,
        count=count,
    )


def _weighted_packing_plan(
    hypergraph: FiniteHypergraph,
    weights: tuple[EdgeWeight, ...],
) -> tuple[
    tuple[str, ...],
    tuple[frozenset[str], ...],
    tuple[Fraction, ...],
    tuple[tuple[int, ...], ...],
    int,
]:
    """Build the reusable conflict-component plan for one weighted packing."""

    edges = _canonical_edges(hypergraph)
    weight_of = {entry.edge_id: entry.weight.as_fraction() for entry in weights}
    search_ids = tuple(edge_id for edge_id, _ in edges)
    edge_sets = tuple(frozenset(members) for _, members in edges)
    values = tuple(weight_of[edge_id] for edge_id in search_ids)
    components = _conflict_components(edge_sets)
    search_work = sum(
        (1 << len(component)) * len(component) * len(component)
        for component in components
    )
    return search_ids, edge_sets, values, components, search_work


def _maximum_component_packing(
    edge_sets: tuple[frozenset[str], ...],
    values: tuple[Fraction, ...],
    search_ids: tuple[str, ...],
    component: tuple[int, ...],
) -> tuple[tuple[str, ...], Fraction]:
    """Return the best disjoint subfamily of one component with its weight.

    Every subset is scored exactly; the best keeps the greatest total
    weight, breaking ties toward the lexicographically smallest
    declared-order family within the component. The union over components
    is the witness: weights add across independent components, while a
    globally smallest ID-tuple need not decompose.
    """

    from itertools import combinations

    best_ids: tuple[str, ...] = ()
    best_weight = Fraction(0)
    members = [edge_sets[index] for index in component]
    worth = [values[index] for index in component]
    names = [search_ids[index] for index in component]
    for size in range(len(component) + 1):
        for combo in combinations(range(len(component)), size):
            disjoint = True
            for left in range(len(combo)):
                for right in range(left + 1, len(combo)):
                    if members[combo[left]] & members[combo[right]]:
                        disjoint = False
                        break
                if not disjoint:
                    break
            if not disjoint:
                continue
            total = sum((worth[position] for position in combo), start=Fraction(0))
            ids = tuple(names[position] for position in combo)
            if total > best_weight or (total == best_weight and ids < best_ids):
                best_ids, best_weight = ids, total
    return best_ids, best_weight


def maximum_weight_packing(
    hypergraph: FiniteHypergraph,
    weights: tuple[EdgeWeight, ...],
) -> WeightedPackingResult:
    """Compute an exact maximum-weight pairwise-disjoint hyperedge family."""

    weight_ids = tuple(entry.edge_id for entry in weights)
    if len(set(weight_ids)) != len(weight_ids):
        raise OperationDomainValidationError(
            location=("weights",),
            code="hypergraph.weighted_packing.weight_identity",
            message="packing weights must use distinct edge IDs",
        )
    edge_ids = tuple(edge_id for edge_id, _ in hypergraph.edges)
    if set(weight_ids) != set(edge_ids):
        raise OperationDomainValidationError(
            location=("weights",),
            code="hypergraph.weighted_packing.weight_coverage",
            message="packing weights must cover exactly the source hyperedge IDs",
        )
    search_ids, edge_sets, values, components, search_work = _weighted_packing_plan(
        hypergraph, weights
    )
    if any(len(component) > MAX_MATCHING_EDGES for component in components):
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.weighted_packing.search_bound",
            message=(
                "maximum weight packing search exceeds the "
                f"{MAX_MATCHING_EDGES}-edge exact search bound"
            ),
        )
    if search_work > MAX_WEIGHTED_PACKING_SEARCH_WORK:
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph.weighted_packing.search_bound",
            message=(
                "maximum weight packing search exceeds the "
                f"{MAX_WEIGHTED_PACKING_SEARCH_WORK:,}-check exact search bound"
            ),
        )
    chosen: set[str] = set()
    total = Fraction(0)
    for component in components:
        best_ids, best_weight = _maximum_component_packing(
            edge_sets, values, search_ids, component
        )
        chosen.update(best_ids)
        total += best_weight
    packing = tuple(edge_id for edge_id in edge_ids if edge_id in chosen)
    return WeightedPackingResult._from_kernel(
        hypergraph=hypergraph,
        weights=weights,
        packing=packing,
        total_weight=CanonicalRational.from_fraction(total),
    )
