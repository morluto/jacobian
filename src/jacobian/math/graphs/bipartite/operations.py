"""Dulmage--Mendelsohn regions via matching, alternating paths and SCCs."""

import time

from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.math.graphs.bipartite._models import DulmageMendelsohnDecomposition
from jacobian.math.graphs.bipartite.values import (
    BipartiteVertexRegion,
    FixedBipartiteGraph,
)

__all__ = ["dulmage_mendelsohn"]


def _reachable(seeds: set[int], adjacency: list[list[int]]) -> set[int]:
    found = set(seeds)
    pending = list(seeds)
    while pending:
        v = pending.pop()
        for w in adjacency[v]:
            if w not in found:
                found.add(w)
                pending.append(w)
    return found


def dulmage_mendelsohn(graph: FixedBipartiteGraph) -> DulmageMendelsohnDecomposition:
    """Return maximum-matching-independent regions, balanced blocks and their DAG.

    left_excess has more left vertices; right_excess has more right vertices.
    Either region can be empty. For matrix patterns, left denotes rows and
    right columns; condensation arcs point along block-upper-triangular order.
    """
    if not isinstance(graph, FixedBipartiteGraph):
        raise TypeError("dulmage_mendelsohn expects FixedBipartiteGraph")
    if current_request_execution() is None:
        with request_execution(time.monotonic()):
            return _decompose(graph)
    return _decompose(graph)


def _decompose(source: FixedBipartiteGraph) -> DulmageMendelsohnDecomposition:
    import networkx as nx

    execution = current_request_execution()
    assert execution is not None
    deadline = execution.started_at + 30.0
    bind_request_deadline(
        min(deadline, execution.deadline)
        if execution.deadline is not None
        else deadline
    )
    request_checkpoint("before Dulmage-Mendelsohn admission")
    n = source.graph.vertex_count
    # Admit the entire canonical 1024-vertex/65,536-edge envelope.
    # Hopcroft--Karp has O((V+E)*sqrt(V)) work; the standard twice-ceil-sqrt
    # phase estimate plus traversals and sorting is at most 6,100,000 units
    # here: (V+E)*(2*(isqrt(V)+1)+bit_length(E+1)+8).
    # SCC/output storage is O(V+E); there are at most E condensation arcs.
    # The recursive matching DFS has at most min(|L|,|R|)+1 <=513 levels.
    # No transitive closure, barrier certificate or matching enumeration runs.
    graph: nx.Graph[int] = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(source.graph.edges)
    # Reuse the maintained bipartite matching backend already used by
    # poset owners. General matching's Tutte--Berge certificate and repeated
    # vertex-deletion solves do not belong to this decomposition postcondition.
    matching: dict[int, int] = nx.algorithms.bipartite.maximum_matching(
        graph, top_nodes=source.left_vertices
    )
    request_checkpoint("after maximum bipartite matching")
    left = set(source.left_vertices)
    arcs: list[list[int]] = [[] for _ in range(n)]
    reverse: list[list[int]] = [[] for _ in range(n)]
    for a, b in source.graph.edges:
        u, v = (a, b) if a in left else (b, a)
        arcs[u].append(v)
        reverse[v].append(u)
        if matching.get(u) == v:
            arcs[v].append(u)
            reverse[u].append(v)
    left_excess = _reachable(left - matching.keys(), arcs)
    right_excess = _reachable(set(source.right_vertices) - matching.keys(), reverse)
    balanced = set(range(n)) - left_excess - right_excess
    request_checkpoint("after alternating reachability")
    digraph: nx.DiGraph[int] = nx.DiGraph()
    digraph.add_nodes_from(sorted(balanced))
    digraph.add_edges_from((v, w) for v in balanced for w in arcs[v] if w in balanced)
    components = list(nx.strongly_connected_components(digraph))
    left_position = {v: i for i, v in enumerate(source.left_vertices)}
    components.sort(
        key=lambda component: min(left_position[v] for v in component if v in left)
    )
    component_of = {v: i for i, component in enumerate(components) for v in component}
    edges = tuple(
        sorted(
            {
                (component_of[v], component_of[w])
                for v, w in digraph.edges
                if component_of[v] != component_of[w]
            }
        )
    )
    request_checkpoint("after balanced SCC condensation")

    right_position = {v: i for i, v in enumerate(source.right_vertices)}

    def region(vertices: set[int]) -> BipartiteVertexRegion:
        return BipartiteVertexRegion(
            left_vertices=tuple(sorted(vertices & left, key=left_position.__getitem__)),
            right_vertices=tuple(
                sorted(vertices - left, key=right_position.__getitem__)
            ),
        )

    result = DulmageMendelsohnDecomposition(
        graph=source,
        structural_rank=len(matching) // 2,
        left_excess=region(left_excess),
        right_excess=region(right_excess),
        balanced_blocks=tuple(region(component) for component in components),
        condensation_edges=edges,
    )
    request_checkpoint("after Dulmage-Mendelsohn result construction")
    return result
