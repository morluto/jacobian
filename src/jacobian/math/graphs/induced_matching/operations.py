"""Maximum induced matching through a private conflict graph."""

from jacobian.math.graphs.independence import (
    IndependenceNumberBudget,
    independence_number,
)
from jacobian.math.graphs.induced_matching._models import (
    MaximumInducedMatchingResult,
    SourceEdgeBinding,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def maximum_induced_matching(
    graph: SimpleUndirectedGraph,
    *,
    resource_budget: IndependenceNumberBudget | None = None,
) -> MaximumInducedMatchingResult:
    """Return an exact or rigorously bounded maximum induced matching."""

    bindings = tuple(
        SourceEdgeBinding(edge_id=f"e{index}", endpoints=edge)
        for index, edge in enumerate(graph.edges)
    )
    source_edge_set = {frozenset(edge) for edge in graph.edges}
    conflicts: list[tuple[str, str]] = []
    for left_index, left in enumerate(bindings):
        left_endpoints = set(left.endpoints)
        for right in bindings[left_index + 1 :]:
            right_endpoints = set(right.endpoints)
            cross_edge = any(
                frozenset((u, v)) in source_edge_set
                for u in left_endpoints
                for v in right_endpoints
            )
            if left_endpoints & right_endpoints or cross_edge:
                conflicts.append((left.edge_id, right.edge_id))
    conflict_graph = SimpleUndirectedGraph(
        vertices=tuple(binding.edge_id for binding in bindings),
        edges=tuple(conflicts),
    )
    independence = independence_number(
        conflict_graph, resource_budget=resource_budget or IndependenceNumberBudget()
    )
    selected_ids = independence.witness_vertices
    selected = {
        endpoint
        for binding in bindings
        if binding.edge_id in selected_ids
        for endpoint in binding.endpoints
    }
    endpoint_graph = SimpleUndirectedGraph(
        vertices=tuple(vertex for vertex in graph.vertices if vertex in selected),
        edges=tuple(
            edge for edge in graph.edges if edge[0] in selected and edge[1] in selected
        ),
    )
    return MaximumInducedMatchingResult(
        graph=graph,
        source_edges=bindings,
        status=independence.status,
        cardinality=independence.incumbent_value,
        lower_bound=independence.lower_bound,
        upper_bound=independence.upper_bound,
        selected_edge_ids=selected_ids,
        induced_endpoint_graph=endpoint_graph,
    )


__all__ = ["maximum_induced_matching"]
