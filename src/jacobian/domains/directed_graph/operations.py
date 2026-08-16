"""Domain adapter for bounded directed-graph operations."""

from __future__ import annotations

from typing import Any

import networkx as nx

from jacobian.contracts.directed_graph import (
    AcyclicOrderRequest,
    AcyclicOrderResult,
    DegreeProfileRequest,
    DegreeProfileResult,
    DirectedGraph,
    ReachabilityRequest,
    ReachabilityResult,
    StrongComponentsRequest,
    StrongComponentsResult,
    TransitiveClosureRequest,
    TransitiveClosureResult,
)


def _build_digraph(graph: DirectedGraph) -> nx.DiGraph[int]:
    """Build a NetworkX digraph from the contract value."""
    g: nx.DiGraph[Any] = nx.DiGraph()
    g.add_nodes_from(range(graph.vertex_count))
    for tail, head in graph.arcs:
        g.add_edge(tail, head)
    return g


def compute_reachability(request: ReachabilityRequest) -> ReachabilityResult:
    """Compute directed reachability from a single source.

    Returns reachable/unreachable vertex sets, minimum directed distance
    from source for each reachable vertex, one deterministic predecessor
    per reachable vertex, and optionally target_reachable + shortest path.
    """
    g = _build_digraph(request.graph)
    source = request.source

    # Use NetworkX BFS predecessors/distance for deterministic results.
    # nx.single_source_shortest_path_length gives BFS distances.
    distances_raw = nx.single_source_shortest_path_length(g, source)

    # Build predecessors via BFS using NetworkX's BFS tree.
    bfs_tree = nx.bfs_tree(g, source)
    predecessors: dict[int, int | None] = {source: None}
    # bfs_tree edges point parent -> child; iterate to find each child's parent.
    for parent, child in bfs_tree.edges():
        predecessors[child] = parent

    reachable_set = set(distances_raw.keys())
    all_vertices = set(range(request.graph.vertex_count))
    unreachable_set = all_vertices - reachable_set

    target_reachable: bool | None = None
    shortest_path: tuple[int, ...] | None = None

    if request.target is not None:
        target_reachable = request.target in reachable_set
        if target_reachable:
            # Reconstruct shortest path from predecessors.
            path: list[int] = []
            current: int | None = request.target
            while current is not None:
                path.append(current)
                current = predecessors.get(current)  # type: ignore[arg-type]
            path.reverse()
            shortest_path = tuple(path)

    return ReachabilityResult(
        source=source,
        reachable=tuple(sorted(reachable_set)),
        unreachable=tuple(sorted(unreachable_set)),
        distances={v: d for v, d in distances_raw.items()},
        predecessors=predecessors,
        target=request.target,
        target_reachable=target_reachable,
        shortest_path=shortest_path,
    )


def compute_strong_components(
    request: StrongComponentsRequest,
) -> StrongComponentsResult:
    """Compute strongly connected components and condensation DAG.

    Returns the canonical SCC partition, component IDs for every vertex,
    the condensation DAG arcs, source/sink components, and a strong
    connectivity boolean.
    """
    g = _build_digraph(request.graph)

    sccs = list(nx.strongly_connected_components(g))

    # Sort components deterministically: by min vertex in each component.
    sccs_sorted = sorted(sccs, key=lambda comp: min(comp))

    component_ids: dict[int, int] = {}
    components: dict[int, tuple[int, ...]] = {}
    for idx, comp in enumerate(sccs_sorted):
        comp_tuple = tuple(sorted(comp))
        components[idx] = comp_tuple
        for v in comp_tuple:
            component_ids[v] = idx

    # Build condensation DAG.
    condensation: set[tuple[int, int]] = set()
    for tail, head in g.edges():
        c_tail = component_ids[tail]
        c_head = component_ids[head]
        if c_tail != c_head:
            condensation.add((c_tail, c_head))

    condensation_arcs = tuple(sorted(condensation))

    # Source components: no incoming arcs in condensation.
    # Sink components: no outgoing arcs in condensation.
    outgoing: set[int] = set()
    incoming: set[int] = set()
    for c_tail, c_head in condensation:
        outgoing.add(c_tail)
        incoming.add(c_head)

    all_component_ids = set(range(len(sccs_sorted)))
    source_components = tuple(sorted(all_component_ids - incoming))
    sink_components = tuple(sorted(all_component_ids - outgoing))

    is_strongly_connected = len(sccs_sorted) <= 1 and request.graph.vertex_count > 0

    return StrongComponentsResult(
        component_count=len(sccs_sorted),
        component_ids=component_ids,
        components=components,
        condensation_arcs=condensation_arcs,
        source_components=source_components,
        sink_components=sink_components,
        is_strongly_connected=is_strongly_connected,
    )


def compute_acyclic_order(request: AcyclicOrderRequest) -> AcyclicOrderResult:
    """Compute a topological order or detect a directed cycle.

    Returns either ACYCLIC with a deterministic topological order and
    position map, or CYCLIC with a concrete directed cycle witness.
    """
    g = _build_digraph(request.graph)

    is_dag = nx.is_directed_acyclic_graph(g)

    if not is_dag:
        cycle = nx.find_cycle(g)
        cycle_nodes = tuple(edge[0] for edge in cycle)
        return AcyclicOrderResult(
            status="CYCLIC",
            cycle_witness=cycle_nodes,
        )

    topo_order = tuple(nx.topological_sort(g))
    positions = {v: i for i, v in enumerate(topo_order)}

    return AcyclicOrderResult(
        status="ACYCLIC",
        topological_order=topo_order,
        positions=positions,
    )


def compute_transitive_closure(
    request: TransitiveClosureRequest,
) -> TransitiveClosureResult:
    """Compute the transitive closure of a directed graph.

    Returns the complete reachable ordered-pair relation.  The reflexive
    convention is explicitly declared: if reflexive=True, (v,v) pairs are
    included for all vertices; otherwise only pairs reachable via at least
    one arc.
    """
    g = _build_digraph(request.graph)

    closure_graph: nx.DiGraph = nx.transitive_closure(g, reflexive=request.reflexive)

    closure_pairs: list[tuple[int, int]] = []
    for u, v in closure_graph.edges():
        closure_pairs.append((u, v))

    closure_pairs.sort()

    return TransitiveClosureResult(
        closure_pairs=tuple(closure_pairs),
        vertex_count=request.graph.vertex_count,
        reflexive=request.reflexive,
    )


def compute_degree_profile(request: DegreeProfileRequest) -> DegreeProfileResult:
    """Compute exact in-degree and out-degree for every vertex.

    Returns in-degrees, out-degrees, sources (in-degree 0), sinks
    (out-degree 0), and isolated vertices (both 0).
    """
    g = _build_digraph(request.graph)

    vertex_count = request.graph.vertex_count
    in_degrees = tuple(dict(g.in_degree()).get(v, 0) for v in range(vertex_count))
    out_degrees = tuple(dict(g.out_degree()).get(v, 0) for v in range(vertex_count))

    sources = tuple(
        sorted(v for v in range(vertex_count) if in_degrees[v] == 0)
    )
    sinks = tuple(
        sorted(v for v in range(vertex_count) if out_degrees[v] == 0)
    )
    isolated = tuple(
        sorted(
            v
            for v in range(vertex_count)
            if in_degrees[v] == 0 and out_degrees[v] == 0
        )
    )

    return DegreeProfileResult(
        in_degrees=in_degrees,
        out_degrees=out_degrees,
        sources=sources,
        sinks=sinks,
        isolated=isolated,
    )
