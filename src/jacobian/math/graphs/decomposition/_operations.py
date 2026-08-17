"""Domain-owned structural graph decomposition operations."""

from __future__ import annotations

import networkx as nx

from jacobian.math.graphs.decomposition._models import (
    BiconnectedComponentsRequest,
    BiconnectedComponentsResult,
    BlockCutTreeRequest,
    BlockCutTreeResult,
    BridgeBlockRequest,
    BridgeBlockResult,
    EarDecompositionRequest,
    EarDecompositionResult,
    UndirectedGraph,
)


def _build_graph(graph: UndirectedGraph) -> nx.Graph:
    """Build a NetworkX undirected graph from the contract model.

    All declared vertices are added as nodes, even isolated ones, so that
    decomposition routines that rely on graph membership observe every
    vertex in the input.
    """
    g = nx.Graph()
    g.add_nodes_from(range(graph.vertex_count))
    for source, target in graph.edges:
        g.add_edge(source, target)
    return g


def compute_block_cut_tree(request: BlockCutTreeRequest) -> BlockCutTreeResult:
    """Compute the block-cut tree decomposition of an undirected graph.

    Uses ``nx.biconnected_components`` to identify the biconnected blocks
    and ``nx.articulation_points`` to identify the cut vertices, then
    constructs the bipartite block-cut tree: an edge connects a block to
    each articulation point it contains.
    """
    g = _build_graph(request.graph)
    blocks = [frozenset(component) for component in nx.biconnected_components(g)]
    articulation_points = sorted(nx.articulation_points(g))

    tree_edges: list[tuple[int, int]] = []
    for block_index, block in enumerate(blocks):
        for vertex in articulation_points:
            if vertex in block:
                tree_edges.append((block_index, vertex))

    return BlockCutTreeResult(
        blocks=tuple(tuple(sorted(block)) for block in blocks),
        articulation_points=tuple(articulation_points),
        tree=tuple(tree_edges),
    )


def compute_bridge_block_tree(request: BridgeBlockRequest) -> BridgeBlockResult:
    """Compute the bridge-block (2-edge-connected component) decomposition.

    Uses ``nx.bridges`` to identify bridges.  Removing all bridges partitions
    the graph into its 2-edge-connected components; the bridge block tree
    connects two components whenever a bridge joins them.
    """
    g = _build_graph(request.graph)
    bridges = list(nx.bridges(g))

    # Contract each non-bridge edge to form the 2-edge-connected components.
    contracted: nx.Graph = nx.Graph()
    contracted.add_nodes_from(g.nodes())
    bridge_set = {(min(u, v), max(u, v)) for u, v in bridges}
    for source, target in g.edges():
        edge = (min(source, target), max(source, target))
        if edge not in bridge_set:
            contracted.add_edge(source, target)

    components = [
        frozenset(component) for component in nx.connected_components(contracted)
    ]
    component_index: dict[int, int] = {}
    for index, component in enumerate(components):
        for vertex in component:
            component_index[vertex] = index

    tree_edges: list[tuple[int, int]] = []
    normalised_bridges: list[tuple[int, int]] = []
    for source, target in bridges:
        normalised_bridges.append((min(source, target), max(source, target)))
        source_component = component_index[source]
        target_component = component_index[target]
        if source_component != target_component:
            tree_edges.append((source_component, target_component))

    return BridgeBlockResult(
        components=tuple(tuple(sorted(component)) for component in components),
        bridges=tuple(normalised_bridges),
        tree=tuple(tree_edges),
    )


def compute_ear_decomposition(
    request: EarDecompositionRequest,
) -> EarDecompositionResult:
    """Compute an open ear decomposition of a biconnected graph.

    NetworkX (3.6) does not expose a public ``ear_decomposition`` function, so
    we implement the standard algorithm:

    1. Starting from an arbitrary vertex, find an initial cycle (the first
       ear).  For a biconnected graph with at least two vertices such a cycle
       always exists.
    2. Iteratively grow the decomposition by finding ears: simple paths whose
       internal vertices are disjoint from the current decomposition, whose
       endpoints lie in the decomposition, and whose edges are unused.

    The result is a sequence of ears ``P_0, P_1, ...`` where ``P_0`` is a cycle
    and each subsequent ear is a path.  This is the canonical open ear
    decomposition guaranteed to exist for any biconnected graph.
    """
    g = _build_graph(request.graph)

    if g.number_of_nodes() < 2:
        return EarDecompositionResult(biconnected=True, ears=())

    if g.number_of_nodes() == 2:
        return EarDecompositionResult(
            biconnected=g.has_edge(0, 1),
            ears=(),
        )

    if not nx.is_biconnected(g):
        return EarDecompositionResult(biconnected=False, ears=())

    # --- First ear: a cycle through the smallest vertex --------------------
    start = min(g.nodes())
    cycle = _find_cycle(g, start)
    used_vertices: set[int] = set(cycle)
    used_edges: set[tuple[int, int]] = set()
    for u, v in zip(cycle, cycle[1:]):  # noqa: RUF007, B905
        used_edges.add((min(u, v), max(u, v)))
    ears: list[tuple[int, ...]] = [tuple(cycle)]

    # --- Subsequent ears ---------------------------------------------------
    while True:
        ear = _find_next_ear(g, used_vertices, used_edges)
        if ear is None:
            break
        ears.append(ear)
        used_vertices.update(ear)
        for u, v in zip(ear, ear[1:]):  # noqa: RUF007, B905
            used_edges.add((min(u, v), max(u, v)))

    return EarDecompositionResult(biconnected=True, ears=tuple(ears))


def _find_cycle(g: nx.Graph[int], start: int) -> list[int]:
    """Return a simple cycle containing ``start`` in ``g``.

    Uses ``nx.find_cycle`` to obtain the cycle edges and reconstructs the
    vertex sequence.  Assumes ``g`` is biconnected with at least two vertices,
    so such a cycle always exists.
    """
    edges = nx.find_cycle(g, source=start)
    cycle: list[int] = []
    for edge in edges:
        u, v = edge[0], edge[1]
        if not cycle:
            cycle.append(u)
        cycle.append(v)
    return cycle


def _find_next_ear(  # noqa: C901
    g: nx.Graph[int],
    used_vertices: set[int],
    used_edges: set[tuple[int, int]],
) -> tuple[int, ...] | None:
    """Find one ear for the open ear decomposition.

    An ear is a simple path (Whitney) whose endpoints are both in
    ``used_vertices``, all internal vertices are unused, and all edges are
    unused.  The BFS parent edge is skipped because reversing the discovery
    edge would be the walk ``s-v-s`` on one edge, which is not a simple path
    and not an ear.  A genuine return to ``s`` must use a different unused
    edge.
    """
    for s in sorted(used_vertices):
        parent: dict[int, int] = {s: s}
        queue: list[int] = [s]
        found: int | None = None
        close_from = s
        while queue and found is None:
            current = queue.pop(0)
            for neighbor in g.neighbors(current):
                edge = (min(current, neighbor), max(current, neighbor))
                if edge in used_edges:
                    continue
                if parent.get(current) == neighbor:
                    continue
                if neighbor in used_vertices:
                    found = neighbor
                    close_from = current
                    if neighbor != s:
                        parent[neighbor] = current
                    break
                if neighbor not in parent:
                    parent[neighbor] = current
                    queue.append(neighbor)
            if found is not None:
                break
        if found is not None:
            ear = _reconstruct_ear(parent, s, found, close_from)
            if ear is not None:
                return ear
    return None


def _reconstruct_ear(
    parent: dict[int, int],
    start: int,
    found: int,
    close_from: int,
) -> tuple[int, ...] | None:
    node: int | None = close_from if found == start else found
    ear: list[int] = []
    while node is not None and node != start:
        ear.append(node)
        node = parent.get(node)
    if node == start:
        ear.append(start)
    ear.reverse()
    if found == start and ear and ear[-1] != start:
        ear.append(start)
    if len(ear) >= 2:
        return tuple(ear)
    return None


def compute_biconnected_components(
    request: BiconnectedComponentsRequest,
) -> BiconnectedComponentsResult:
    """List all biconnected components of an undirected graph.

    Uses ``nx.biconnected_components`` directly.
    """
    g = _build_graph(request.graph)
    components = list(nx.biconnected_components(g))
    return BiconnectedComponentsResult(
        components=tuple(tuple(sorted(component)) for component in components),
    )
