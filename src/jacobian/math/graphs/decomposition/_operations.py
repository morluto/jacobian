"""Domain-owned structural graph decomposition operations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal, cast

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
    SPQRSkeleton,
    SPQRTreeRequest,
    SPQRTreeResult,
    UndirectedGraph,
)
from jacobian.math.graphs.multigraph._models import LooplessMultigraph, MultigraphEdge


def _build_graph(graph: UndirectedGraph) -> nx.Graph[int]:
    """Build a NetworkX undirected graph from the contract model.

    All declared vertices are added as nodes, even isolated ones, so that
    decomposition routines that rely on graph membership observe every
    vertex in the input.
    """
    g: nx.Graph[int] = nx.Graph()
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
    blocks = [
        frozenset(cast(set[int], component))
        for component in nx.biconnected_components(g)
    ]
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
    bridges = list(cast(list[tuple[int, int]], nx.bridges(g)))

    # Contract each non-bridge edge to form the 2-edge-connected components.
    contracted: nx.Graph[int] = nx.Graph()
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


def _find_next_ear(
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
    components = [cast(set[int], c) for c in nx.biconnected_components(g)]
    return BiconnectedComponentsResult(
        components=tuple(tuple(sorted(component)) for component in components),
    )


# ---------------------------------------------------------------------------
# Bounded normalized SPQR construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SPQREdge:
    edge_id: str
    left: int
    right: int
    source_edge: tuple[int, int] | None

    @property
    def endpoints(self) -> tuple[int, int]:
        return (min(self.left, self.right), max(self.left, self.right))

    @property
    def is_virtual(self) -> bool:
        return self.source_edge is None


@dataclass(frozen=True)
class _Anchor:
    node_id: str
    edge_id: str


type _SPQRKind = Literal["S_NODE", "P_NODE", "Q_NODE", "R_NODE"]


def _as_public_skeleton(
    node_id: str, kind: _SPQRKind, edges: tuple[_SPQREdge, ...]
) -> SPQRSkeleton:
    vertices = tuple(
        sorted({vertex for edge in edges for vertex in (edge.left, edge.right)})
    )
    positions = {vertex: index for index, vertex in enumerate(vertices)}
    return SPQRSkeleton(
        node_id=node_id,
        kind=kind,
        vertices=vertices,
        graph=LooplessMultigraph(
            vertex_count=len(vertices),
            edges=tuple(
                MultigraphEdge(
                    edge_id=edge.edge_id,
                    left=positions[edge.left],
                    right=positions[edge.right],
                )
                for edge in edges
            ),
        ),
        real_edge_sources=tuple(
            (edge.edge_id, edge.source_edge)
            for edge in edges
            if edge.source_edge is not None
        ),
        virtual_edge_ids=tuple(edge.edge_id for edge in edges if edge.is_virtual),
    )


class _SPQRBuilder:
    """Deterministic split-component SPQR construction for one small graph.

    This deliberately uses NetworkX only for connectedness/biconnectedness.
    The split-component recursion owns separator choice, virtual-edge pairing,
    S/P normalization, and the source-edge transport.  A boundary virtual
    edge is retained while descending a child; when its endpoints are itself a
    separation pair it becomes an edge of the new P skeleton.  That detail is
    what prevents a recursive split from losing the parent's gluing interface.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, tuple[_SPQRKind, tuple[_SPQREdge, ...]]] = {}
        self._pairs: dict[str, str] = {}
        self._tree_edges: list[tuple[str, str]] = []
        self._next_node = 0
        self._next_virtual = 0

    def build(self, graph: UndirectedGraph) -> SPQRTreeResult:
        source_edges = tuple(
            _SPQREdge(
                edge_id=f"real:{min(left, right)}:{max(left, right)}",
                left=min(left, right),
                right=max(left, right),
                source_edge=(min(left, right), max(left, right)),
            )
            for left, right in sorted(graph.edges)
        )
        self._decompose(source_edges, boundary_edge_id=None)
        self._normalize_adjacent_nodes()
        nodes = tuple(
            _as_public_skeleton(node_id, kind, edges)
            for node_id, (kind, edges) in sorted(self._nodes.items())
        )
        owners = tuple(
            sorted(
                (
                    edge.source_edge,
                    node_id,
                    edge.edge_id,
                )
                for node_id, (_, edges) in self._nodes.items()
                for edge in edges
                if edge.source_edge is not None
            )
        )
        vertex_incidence = tuple(
            (
                vertex,
                tuple(
                    node_id
                    for node_id, (_, edges) in sorted(self._nodes.items())
                    if any(vertex in (edge.left, edge.right) for edge in edges)
                ),
            )
            for vertex in range(graph.vertex_count)
        )
        pairs = tuple(
            sorted((left, right) for left, right in self._pairs.items() if left < right)
        )
        return SPQRTreeResult._from_kernel(
            source_graph=graph,
            status="SPQR_TREE",
            nodes=nodes,
            tree_edges=tuple(
                sorted(
                    (min(left, right), max(left, right))
                    for left, right in self._tree_edges
                )
            ),
            virtual_edge_pairs=pairs,
            source_vertex_incidence=vertex_incidence,
            source_edge_owners=owners,
        )

    def _new_virtual(self, left: int, right: int) -> _SPQREdge:
        edge = _SPQREdge(
            edge_id=f"virtual:{self._next_virtual}",
            left=min(left, right),
            right=max(left, right),
            source_edge=None,
        )
        self._next_virtual += 1
        return edge

    def _new_node(self, kind: _SPQRKind, edges: tuple[_SPQREdge, ...]) -> str:
        node_id = f"node:{self._next_node}"
        self._next_node += 1
        self._nodes[node_id] = (
            kind,
            tuple(sorted(edges, key=lambda edge: edge.edge_id)),
        )
        return node_id

    def _pair(self, left: _Anchor, right: _Anchor) -> None:
        if left.edge_id in self._pairs or right.edge_id in self._pairs:
            raise ValueError("a virtual SPQR edge may have only one mate")
        self._pairs[left.edge_id] = right.edge_id
        self._pairs[right.edge_id] = left.edge_id
        self._tree_edges.append((left.node_id, right.node_id))

    def _normalize_adjacent_nodes(self) -> None:
        """Merge every adjacent S/S or P/P pair through its virtual join."""
        while True:
            locations = {
                edge.edge_id: node_id
                for node_id, (_, edges) in self._nodes.items()
                for edge in edges
            }
            merge: tuple[str, str, str, str] | None = None
            for left_edge, right_edge in self._pairs.items():
                if left_edge > right_edge:
                    continue
                left_node, right_node = locations[left_edge], locations[right_edge]
                if left_node == right_node:
                    continue
                left_kind = self._nodes[left_node][0]
                if left_kind == self._nodes[right_node][0] and left_kind in {
                    "S_NODE",
                    "P_NODE",
                }:
                    merge = (left_node, right_node, left_edge, right_edge)
                    break
            if merge is None:
                return
            self._merge_nodes(*merge)

    def _merge_nodes(
        self,
        left_node: str,
        right_node: str,
        left_edge_id: str,
        right_edge_id: str,
    ) -> None:
        kind, left_edges = self._nodes[left_node]
        _, right_edges = self._nodes[right_node]
        merged_edges = tuple(
            edge
            for edge in (*left_edges, *right_edges)
            if edge.edge_id not in {left_edge_id, right_edge_id}
        )
        self._nodes[left_node] = (
            kind,
            tuple(sorted(merged_edges, key=lambda edge: edge.edge_id)),
        )
        del self._nodes[right_node]
        del self._pairs[left_edge_id]
        del self._pairs[right_edge_id]
        normalized: set[tuple[str, str]] = set()
        for source, target in self._tree_edges:
            source = left_node if source == right_node else source
            target = left_node if target == right_node else target
            if source != target:
                normalized.add((min(source, target), max(source, target)))
        self._tree_edges = sorted(normalized)

    def _decompose(
        self,
        edges: tuple[_SPQREdge, ...],
        boundary_edge_id: str | None,
    ) -> _Anchor | None:
        base_kind = self._base_kind(edges)
        split = (
            None
            if base_kind in {"S_NODE", "P_NODE", "Q_NODE"}
            else self._first_split(edges, boundary_edge_id)
        )
        if split is None:
            node_id = self._new_node(base_kind, edges)
            if boundary_edge_id is None:
                return None
            if boundary_edge_id not in {edge.edge_id for edge in edges}:
                raise ValueError("boundary virtual edge was lost during SPQR recursion")
            return _Anchor(node_id, boundary_edge_id)

        left, right, fragments, direct_real, direct_virtual = split
        children: list[_Anchor] = []
        for fragment in fragments:
            child_boundary = self._new_virtual(left, right)
            child = self._decompose((*fragment, child_boundary), child_boundary.edge_id)
            if child is None:
                raise ValueError("a split component must retain its boundary edge")
            children.append(child)
        for edge in direct_real:
            child_boundary = self._new_virtual(left, right)
            node_id = self._new_node("Q_NODE", (edge, child_boundary))
            children.append(_Anchor(node_id, child_boundary.edge_id))

        if direct_virtual:
            p_edges = [
                *direct_virtual,
                *(self._new_virtual(left, right) for _ in children),
            ]
            node_id = self._new_node("P_NODE", tuple(p_edges))
            for child, parent_edge in zip(
                children, p_edges[len(direct_virtual) :], strict=True
            ):
                self._pair(_Anchor(node_id, parent_edge.edge_id), child)
            if boundary_edge_id in {edge.edge_id for edge in direct_virtual}:
                return _Anchor(node_id, boundary_edge_id)
            return self._anchor_for_virtual(boundary_edge_id)

        if len(children) == 2:
            self._pair(children[0], children[1])
            return self._anchor_for_virtual(boundary_edge_id)
        if len(children) < 3:
            raise ValueError("a non-boundary split must have at least two components")
        p_virtual_edges = tuple(self._new_virtual(left, right) for _ in children)
        node_id = self._new_node("P_NODE", p_virtual_edges)
        for child, parent_edge in zip(children, p_virtual_edges, strict=True):
            self._pair(_Anchor(node_id, parent_edge.edge_id), child)
        return self._anchor_for_virtual(boundary_edge_id)

    def _anchor_for_virtual(self, edge_id: str | None) -> _Anchor | None:
        if edge_id is None:
            return None
        for node_id, (_, edges) in self._nodes.items():
            if edge_id in {edge.edge_id for edge in edges}:
                return _Anchor(node_id, edge_id)
        raise ValueError("boundary virtual edge was lost during SPQR recursion")

    def _first_split(
        self,
        edges: tuple[_SPQREdge, ...],
        boundary_edge_id: str | None,
    ) -> (
        tuple[
            int,
            int,
            tuple[tuple[_SPQREdge, ...], ...],
            tuple[_SPQREdge, ...],
            tuple[_SPQREdge, ...],
        ]
        | None
    ):
        vertices = tuple(
            sorted({vertex for edge in edges for vertex in (edge.left, edge.right)})
        )
        for left, right in combinations(vertices, 2):
            removed = {left, right}
            graph: nx.Graph[int] = nx.Graph()
            graph.add_nodes_from(vertex for vertex in vertices if vertex not in removed)
            for edge in edges:
                if edge.left not in removed and edge.right not in removed:
                    graph.add_edge(edge.left, edge.right)
            components = tuple(
                sorted(
                    (
                        frozenset(component)
                        for component in nx.connected_components(graph)
                    ),
                    key=lambda component: tuple(sorted(component)),
                )
            )
            by_component: list[list[_SPQREdge]] = [[] for _ in components]
            direct_real: list[_SPQREdge] = []
            direct_virtual: list[_SPQREdge] = []
            for edge in edges:
                if edge.endpoints == (left, right):
                    if edge.is_virtual:
                        direct_virtual.append(edge)
                    else:
                        direct_real.append(edge)
                    continue
                endpoint = edge.left if edge.left not in removed else edge.right
                for index, component in enumerate(components):
                    if endpoint in component:
                        by_component[index].append(edge)
                        break
                else:
                    raise ValueError(
                        "split component edge did not retain a non-separator endpoint"
                    )
            fragments = tuple(
                tuple(sorted(fragment, key=lambda edge: edge.edge_id))
                for fragment in by_component
                if fragment
            )
            # A separation pair is determined by disconnectedness after its
            # deletion. A direct edge between the pair is a split component
            # only after that condition holds; otherwise every K4 edge would
            # spuriously create a P node.
            if len(fragments) >= 2:
                if direct_virtual:
                    return (
                        left,
                        right,
                        fragments,
                        tuple(sorted(direct_real, key=lambda edge: edge.edge_id)),
                        tuple(sorted(direct_virtual, key=lambda edge: edge.edge_id)),
                    )
                return (
                    left,
                    right,
                    fragments,
                    tuple(sorted(direct_real, key=lambda edge: edge.edge_id)),
                    (),
                )
        return None

    @staticmethod
    def _base_kind(edges: tuple[_SPQREdge, ...]) -> _SPQRKind:
        real = tuple(edge for edge in edges if not edge.is_virtual)
        vertices = {vertex for edge in edges for vertex in (edge.left, edge.right)}
        if len(real) == 1 and len(edges) <= 2:
            return "Q_NODE"
        if len(vertices) == 2 and len(edges) >= 3:
            return "P_NODE"
        degree: dict[int, int] = dict.fromkeys(vertices, 0)
        for edge in edges:
            degree[edge.left] += 1
            degree[edge.right] += 1
        if (
            len(vertices) >= 3
            and len(edges) == len(vertices)
            and all(value == 2 for value in degree.values())
        ):
            return "S_NODE"
        return "R_NODE"


def _negative_spqr_result(graph: UndirectedGraph) -> SPQRTreeResult:
    if graph.vertex_count < 3:
        return SPQRTreeResult._from_kernel(
            source_graph=graph,
            status="NOT_BICONNECTED",
            witness_kind="MINIMUM_SIZE",
            witness_vertices=(0,),
        )
    value = _build_graph(graph)
    components = tuple(
        sorted(nx.connected_components(value), key=lambda component: min(component))
    )
    if len(components) > 1:
        return SPQRTreeResult._from_kernel(
            source_graph=graph,
            status="NOT_BICONNECTED",
            witness_kind="DISCONNECTED",
            witness_vertices=(min(components[0]), min(components[1])),
        )
    articulation = min(nx.articulation_points(value), default=None)
    if articulation is None:
        raise ValueError("positive SPQR precondition failed without a graph witness")
    return SPQRTreeResult._from_kernel(
        source_graph=graph,
        status="NOT_BICONNECTED",
        witness_kind="ARTICULATION",
        witness_vertices=(articulation,),
    )


def compute_spqr_tree(request: SPQRTreeRequest) -> SPQRTreeResult:
    """Compute a deterministic normalized full SPQR tree.

    The producer enumerates bounded vertex-pair split components, inserts
    paired virtual edges, and collapses every degree-two P junction.  It does
    not use NetworkX as an SPQR backend: NetworkX only supplies the initial
    biconnectivity classification and connected-component primitive.
    """
    graph = request.graph
    value = _build_graph(graph)
    if graph.vertex_count < 3 or not nx.is_biconnected(value):
        return _negative_spqr_result(graph)
    return _SPQRBuilder().build(graph)


def _validate_spqr_tree(result: SPQRTreeResult) -> None:
    """Replay the defining SPQR source, tree, and skeleton invariants."""
    if result.status == "NOT_BICONNECTED":
        _validate_negative_spqr_result(result)
        return
    source_value = _build_graph(result.source_graph)
    if result.source_graph.vertex_count < 3 or not nx.is_biconnected(source_value):
        raise ValueError(
            "an SPQR tree requires a biconnected source graph of at least three vertices"
        )
    nodes = {node.node_id: node for node in result.nodes}
    if not nodes:
        raise ValueError("an SPQR tree must contain at least one skeleton")
    if len(nodes) != len(result.nodes):
        raise ValueError("SPQR node IDs must be unique")
    edge_locations, real_sources = _collect_spqr_edges(result)
    source = {tuple(sorted(edge)) for edge in result.source_graph.edges}
    if set(real_sources) != source or len(real_sources) != len(source):
        raise ValueError(
            "real skeleton edges must reconstruct the source edge set exactly"
        )
    expected_owners = tuple(
        sorted(
            (_real_source(node, edge), node_id, edge.edge_id)
            for node_id, node, edge in edge_locations.values()
            if edge.edge_id not in node.virtual_edge_ids
        )
    )
    if result.source_edge_owners != expected_owners:
        raise ValueError("source edge ownership must match the skeleton real edges")
    expected_incidence = tuple(
        (
            vertex,
            tuple(
                node_id
                for node_id, node in sorted(nodes.items())
                if vertex in node.vertices
            ),
        )
        for vertex in range(result.source_graph.vertex_count)
    )
    if result.source_vertex_incidence != expected_incidence:
        raise ValueError("source vertex incidence must match the skeleton carriers")
    derived_tree = _validate_virtual_pairs(result, edge_locations, source_value)
    if tuple(sorted(derived_tree)) != result.tree_edges:
        raise ValueError("SPQR tree edges must be induced by virtual-edge pairs")
    _validate_normalized_tree(nodes, result.tree_edges)


def verify_spqr_tree_result(result: SPQRTreeResult) -> bool:
    """Independently replay one bounded, externally supplied SPQR claim.

    Construction trusts the owner-local decomposition kernel.  This verifier
    is the separate, explicit path for callers that need to establish the
    defining source, skeleton, and normalized-tree invariants of supplied
    serialized data.
    """

    try:
        _validate_spqr_tree(result)
    except (ValueError, nx.NetworkXException):
        return False
    return True


def _collect_spqr_edges(
    result: SPQRTreeResult,
) -> tuple[dict[str, tuple[str, SPQRSkeleton, MultigraphEdge]], list[tuple[int, int]]]:
    edge_locations: dict[str, tuple[str, SPQRSkeleton, MultigraphEdge]] = {}
    real_sources: list[tuple[int, int]] = []
    for node in result.nodes:
        for edge in node.graph.edges:
            if edge.edge_id in edge_locations:
                raise ValueError("a skeleton edge must occur in exactly one node")
            edge_locations[edge.edge_id] = (node.node_id, node, edge)
            if edge.edge_id not in node.virtual_edge_ids:
                real_sources.append(_real_source(node, edge))
        _validate_skeleton_kind(node)
    return edge_locations, real_sources


def _validate_virtual_pairs(
    result: SPQRTreeResult,
    edge_locations: dict[str, tuple[str, SPQRSkeleton, MultigraphEdge]],
    source_value: nx.Graph[int],
) -> set[tuple[str, str]]:
    pair_edges: set[str] = set()
    derived_tree: set[tuple[str, str]] = set()
    separated_pairs: set[tuple[int, int]] = set()
    for left_id, right_id in result.virtual_edge_pairs:
        if left_id == right_id or left_id in pair_edges or right_id in pair_edges:
            raise ValueError("virtual edges must occur in exactly one distinct pair")
        try:
            left_node, left_skeleton, left = edge_locations[left_id]
            right_node, right_skeleton, right = edge_locations[right_id]
        except KeyError as exc:
            raise ValueError("virtual pair references an absent skeleton edge") from exc
        if (
            left_id not in left_skeleton.virtual_edge_ids
            or right_id not in right_skeleton.virtual_edge_ids
        ):
            raise ValueError("only virtual skeleton edges may be paired")
        separator = _global_endpoints(left_skeleton, left)
        if separator != _global_endpoints(right_skeleton, right):
            raise ValueError(
                "paired virtual edges must share their separator endpoints"
            )
        if separator not in separated_pairs:
            remaining = source_value.copy()
            remaining.remove_nodes_from(separator)
            if nx.is_connected(remaining):
                raise ValueError(
                    "paired virtual edges must represent a genuine separation"
                    " pair of the source graph"
                )
            separated_pairs.add(separator)
        if left_node == right_node:
            raise ValueError("paired virtual edges must join distinct SPQR nodes")
        tree_edge = (min(left_node, right_node), max(left_node, right_node))
        if tree_edge in derived_tree:
            raise ValueError(
                "an SPQR tree edge must be glued by exactly one virtual-edge pair"
            )
        pair_edges.update((left_id, right_id))
        derived_tree.add(tree_edge)
    all_virtual = {
        edge_id
        for edge_id, (_, node, _) in edge_locations.items()
        if edge_id in node.virtual_edge_ids
    }
    if pair_edges != all_virtual:
        raise ValueError("every virtual skeleton edge must have exactly one mate")
    return derived_tree


def _validate_normalized_tree(
    nodes: dict[str, SPQRSkeleton], tree_edges: tuple[tuple[str, str], ...]
) -> None:
    tree: nx.Graph[str] = nx.Graph()
    tree.add_nodes_from(nodes)
    tree.add_edges_from(tree_edges)
    if not nx.is_tree(tree):
        raise ValueError("SPQR node graph must be a tree")
    for left_id, right_id in tree_edges:
        left, right = nodes[left_id], nodes[right_id]
        if left.kind == right.kind and left.kind in {"S_NODE", "P_NODE"}:
            raise ValueError(
                "normalized SPQR trees may not have adjacent mergeable S/S or P/P nodes"
            )


def _global_endpoints(node: SPQRSkeleton, edge: MultigraphEdge) -> tuple[int, int]:
    left, right = node.vertices[edge.left], node.vertices[edge.right]
    return (min(left, right), max(left, right))


def _real_source(node: SPQRSkeleton, edge: MultigraphEdge) -> tuple[int, int]:
    for edge_id, source_edge in node.real_edge_sources:
        if edge_id == edge.edge_id:
            return source_edge
    raise ValueError("real skeleton edge missing source map")


def _validate_negative_spqr_result(result: SPQRTreeResult) -> None:
    graph = result.source_graph
    if result.witness_kind == "MINIMUM_SIZE":
        if graph.vertex_count >= 3 or result.witness_vertices != (0,):
            raise ValueError(
                "minimum-size witness must prove the published SPQR convention"
            )
        return
    value = _build_graph(graph)
    if result.witness_kind == "DISCONNECTED":
        if len(result.witness_vertices) != 2:
            raise ValueError("disconnectedness requires two concrete vertices")
        left, right = result.witness_vertices
        if not (0 <= left < graph.vertex_count and 0 <= right < graph.vertex_count):
            raise ValueError(
                "disconnectedness witness vertices must name source vertices"
            )
        if left == right or nx.has_path(value, left, right):
            raise ValueError("disconnectedness witness vertices must be disconnected")
        return
    if result.witness_kind == "ARTICULATION":
        if len(result.witness_vertices) != 1 or result.witness_vertices[0] not in set(
            nx.articulation_points(value)
        ):
            raise ValueError(
                "articulation witness must be an articulation vertex of the source"
            )
        return
    raise ValueError("non-biconnected result has an unknown witness kind")


def _validate_skeleton_kind(node: SPQRSkeleton) -> None:
    degrees: dict[int, int] = dict.fromkeys(node.vertices, 0)
    for edge in node.graph.edges:
        degrees[node.vertices[edge.left]] += 1
        degrees[node.vertices[edge.right]] += 1
    if node.kind == "Q_NODE":
        if (
            len(node.vertices) != 2
            or len(node.real_edge_sources) != 1
            or len(node.graph.edges) > 2
            or any(
                _global_endpoints(node, edge) != node.vertices
                for edge in node.graph.edges
            )
        ):
            raise ValueError(
                "a Q skeleton must carry one source real edge on its two carriers"
            )
        return
    if node.kind == "P_NODE":
        if (
            len(node.vertices) != 2
            or len(node.graph.edges) < 3
            or len(node.virtual_edge_ids) != len(node.graph.edges)
        ):
            raise ValueError(
                "a P skeleton must be a bond of at least three virtual edges"
            )
        return
    if node.kind == "S_NODE":
        skeleton_graph: nx.Graph[int] = nx.Graph()
        skeleton_graph.add_nodes_from(node.vertices)
        skeleton_graph.add_edges_from(
            _global_endpoints(node, edge) for edge in node.graph.edges
        )
        if (
            len(node.vertices) < 3
            or len(node.graph.edges) != len(node.vertices)
            or any(value != 2 for value in degrees.values())
            or not nx.is_connected(skeleton_graph)
        ):
            raise ValueError("an S skeleton must be one cycle")
        return
    endpoint_pairs = [_global_endpoints(node, edge) for edge in node.graph.edges]
    if len(set(endpoint_pairs)) != len(endpoint_pairs):
        raise ValueError("an R skeleton must not repeat an endpoint pair")
    graph: nx.Graph[int] = nx.Graph()
    graph.add_nodes_from(node.vertices)
    graph.add_edges_from(endpoint_pairs)
    if len(node.vertices) < 4 or not nx.is_biconnected(graph):
        raise ValueError("an R skeleton must be triconnected")
    for left, right in combinations(node.vertices, 2):
        remaining = graph.copy()
        remaining.remove_nodes_from((left, right))
        if remaining.number_of_nodes() > 1 and not nx.is_connected(remaining):
            raise ValueError("an R skeleton must have no separation pair")
