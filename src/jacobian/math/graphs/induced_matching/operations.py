"""Maximum induced matching through a private conflict graph."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.independence import (
    IndependenceNumberBudget,
    independence_number,
)
from jacobian.math.graphs.induced_matching._models import (
    MAX_INDUCED_MATCHING_CONFLICT_GRAPH_CELLS,
    MAX_INDUCED_MATCHING_CONFLICT_PAIRS,
    MAX_INDUCED_MATCHING_EDGES,
    MaximumInducedMatchingResult,
    SourceEdgeBinding,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


@dataclass(frozen=True, slots=True)
class _InducedMatchingPlan:
    """Canonical source axis and one materialized private conflict graph."""

    canonical_edges: tuple[tuple[str, str], ...]
    bindings: tuple[SourceEdgeBinding, ...]
    conflict_edges: tuple[tuple[str, str], ...]


def _admit_and_build_plan(
    graph: SimpleUndirectedGraph, budget: IndependenceNumberBudget
) -> _InducedMatchingPlan:
    if not isinstance(graph, SimpleUndirectedGraph):
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.induced_matching.graph_type",
            message="graph must be a SimpleUndirectedGraph",
        )

    # The edge axis is the source identity for this operation. Canonicalize
    # before assigning IDs so input row order is never part of the witness.
    canonical_edges = tuple(sorted(graph.edges))
    edge_count = len(canonical_edges)
    if edge_count > MAX_INDUCED_MATCHING_EDGES:
        raise OperationResourceAdmissionError(
            location=("graph", "edges"),
            code="graph.induced_matching.source_edge_bound",
            message="source edge count exceeds the admitted induced-matching bound",
        )
    if edge_count > budget.max_order:
        raise OperationResourceAdmissionError(
            location=("graph", "edges"),
            code="graph.induced_matching.conflict_graph_order_bound",
            message="source edge count exceeds the admitted conflict-graph order",
        )

    # The complete conflict-pair envelope is C(m, 2), independent of labels.
    # Check it before materializing any pair rows.
    conflict_pair_bound = edge_count * (edge_count - 1) // 2
    if conflict_pair_bound > MAX_INDUCED_MATCHING_CONFLICT_PAIRS:
        raise OperationResourceAdmissionError(
            location=("graph", "edges"),
            code="graph.induced_matching.conflict_pair_bound",
            message="conflict-pair construction exceeds its admitted work envelope",
        )
    if edge_count + conflict_pair_bound > MAX_INDUCED_MATCHING_CONFLICT_GRAPH_CELLS:
        raise OperationResourceAdmissionError(
            location=("graph", "edges"),
            code="graph.induced_matching.conflict_graph_cell_bound",
            message="conflict graph cells exceed the admitted intermediate envelope",
        )
    bindings = tuple(
        SourceEdgeBinding(edge_id=f"e{index}", endpoints=edge)
        for index, edge in enumerate(canonical_edges)
    )
    source_edge_set = {frozenset(edge) for edge in canonical_edges}
    conflicts: list[tuple[str, str]] = []
    for left_index, left in enumerate(bindings):
        request_checkpoint("during induced-matching conflict construction")
        left_endpoints = set(left.endpoints)
        for right in bindings[left_index + 1 :]:
            right_endpoints = set(right.endpoints)
            cross_edge = any(
                frozenset((u, v)) in source_edge_set
                for u in left_endpoints
                for v in right_endpoints
            )
            if left_endpoints & right_endpoints or cross_edge:
                conflicts.append(
                    (left.edge_id, right.edge_id)
                    if left.edge_id < right.edge_id
                    else (right.edge_id, left.edge_id)
                )
    return _InducedMatchingPlan(
        canonical_edges=canonical_edges,
        bindings=bindings,
        conflict_edges=tuple(conflicts),
    )


def maximum_induced_matching(
    graph: SimpleUndirectedGraph,
    *,
    resource_budget: IndependenceNumberBudget | None = None,
) -> MaximumInducedMatchingResult:
    """Return an exact or rigorously bounded maximum induced matching."""

    budget = resource_budget or IndependenceNumberBudget()
    plan = _admit_and_build_plan(graph, budget)
    conflict_graph = SimpleUndirectedGraph(
        # The independence kernel's canonical tie-break follows this axis;
        # sort IDs lexicographically to match the public selected-ID family.
        vertices=tuple(sorted(binding.edge_id for binding in plan.bindings)),
        edges=plan.conflict_edges,
    )
    independence = independence_number(conflict_graph, resource_budget=budget)
    selected_ids = independence.witness_vertices
    selected = {
        endpoint
        for binding in plan.bindings
        if binding.edge_id in selected_ids
        for endpoint in binding.endpoints
    }
    endpoint_graph = SimpleUndirectedGraph(
        vertices=tuple(sorted(selected)),
        edges=tuple(
            edge
            for edge in plan.canonical_edges
            if edge[0] in selected and edge[1] in selected
        ),
    )
    return MaximumInducedMatchingResult(
        graph=graph,
        source_edges=plan.bindings,
        status=independence.status,
        cardinality=independence.incumbent_value,
        lower_bound=independence.lower_bound,
        upper_bound=independence.upper_bound,
        selected_edge_ids=selected_ids,
        induced_endpoint_graph=endpoint_graph,
    )


__all__ = ["maximum_induced_matching"]
