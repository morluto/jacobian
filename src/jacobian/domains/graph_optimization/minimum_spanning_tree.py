"""Exact rational minimum-spanning-tree computation."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.capabilities import (
    CapabilityDiagnostic,
    CapabilityInvocationExample,
    CapabilityMode,
)
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.graph_optimization import (
    CanonicalWeightedTreeEdge,
    GraphMinimumSpanningTreeRequest,
    GraphMinimumSpanningTreeResult,
    GraphMstCycleCheck,
    GraphMstOptimalityCertificate,
)
from jacobian.operations import (
    ComputedNotApplicable,
    ComputedOperation,
    ComputedOutcome,
    ComputedSuccess,
)

if TYPE_CHECKING:
    import networkx as nx

_INVALID_REQUEST = CapabilityDiagnostic(
    code="INVALID_MINIMUM_SPANNING_TREE_REQUEST",
    stage="minimum_spanning_tree_input_validation",
    message=(
        "Input does not satisfy the bounded exact rational weighted-graph contract."
    ),
    hint=(
        "Supply at most 32 unique labelled vertices and at most one exact rational "
        "weight for each simple undirected edge."
    ),
)


def _rational(value: Fraction) -> CanonicalRational:
    return CanonicalRational(
        num=format_canonical_integer(value.numerator),
        den=format_canonical_integer(value.denominator),
    )


def _canonical_endpoints(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _canonical_components(graph: nx.Graph[str]) -> tuple[tuple[str, ...], ...]:
    import networkx as nx

    return tuple(
        sorted(
            (tuple(sorted(component)) for component in nx.connected_components(graph)),
            key=lambda component: component[0],
        )
    )


def compute_minimum_spanning_tree(
    request: GraphMinimumSpanningTreeRequest,
) -> GraphMinimumSpanningTreeResult:
    """Compute one deterministic exact MST and its cycle-property certificate."""

    import networkx as nx

    graph: nx.Graph[str] = nx.Graph()
    graph.add_nodes_from(sorted(request.graph.vertices))
    ordered_edges = sorted(
        request.graph.edges,
        key=lambda edge: (
            edge.weight.as_fraction(),
            *_canonical_endpoints(*edge.endpoints),
        ),
    )
    source_weights: dict[tuple[str, str], Fraction] = {}
    for edge in ordered_edges:
        endpoints = _canonical_endpoints(*edge.endpoints)
        weight = edge.weight.as_fraction()
        source_weights[endpoints] = weight
        graph.add_edge(*endpoints, weight=weight)

    vertices = tuple(sorted(graph))
    components = _canonical_components(graph)
    if not vertices or len(components) != 1:
        return GraphMinimumSpanningTreeResult(
            status="NO_SPANNING_TREE",
            vertices=vertices,
            order=len(vertices),
            connected=False,
            component_count=len(components),
            components=components,
            tree_edges=(),
            total_weight=None,
            optimality_certificate=GraphMstOptimalityCertificate(checks=()),
        )

    tree: nx.Graph[str] = nx.minimum_spanning_tree(
        graph,
        algorithm="kruskal",
        weight="weight",
    )
    selected = {
        _canonical_endpoints(str(left), str(right)) for left, right in tree.edges
    }
    tree_edges = tuple(
        CanonicalWeightedTreeEdge(
            endpoints=endpoints,
            weight=_rational(source_weights[endpoints]),
        )
        for endpoints in sorted(selected)
    )
    checks: list[GraphMstCycleCheck] = []
    for endpoints in sorted(set(source_weights) - selected):
        left, right = endpoints
        path = tuple(nx.shortest_path(tree, left, right))
        path_weights = tuple(
            source_weights[_canonical_endpoints(path[index], path[index + 1])]
            for index in range(len(path) - 1)
        )
        maximum_path_weight = max(path_weights)
        if source_weights[endpoints] < maximum_path_weight:
            raise RuntimeError(
                "maintained MST backend returned a cycle-improvable spanning tree"
            )
        checks.append(
            GraphMstCycleCheck(
                non_tree_edge=endpoints,
                edge_weight=_rational(source_weights[endpoints]),
                tree_path_vertices=path,
                maximum_tree_path_weight=_rational(maximum_path_weight),
            )
        )
    return GraphMinimumSpanningTreeResult(
        status="EXACT",
        vertices=vertices,
        order=len(vertices),
        connected=True,
        component_count=1,
        components=components,
        tree_edges=tree_edges,
        total_weight=_rational(
            sum(
                (source_weights[edge.endpoints] for edge in tree_edges),
                start=Fraction(),
            )
        ),
        optimality_certificate=GraphMstOptimalityCertificate(checks=tuple(checks)),
    )


def _execute(
    request: GraphMinimumSpanningTreeRequest,
) -> ComputedOutcome[GraphMinimumSpanningTreeResult]:
    import networkx as nx

    try:
        return ComputedSuccess(compute_minimum_spanning_tree(request))
    except (
        ArithmeticError,
        nx.NetworkXError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        return ComputedNotApplicable(
            CapabilityDiagnostic(
                code="MINIMUM_SPANNING_TREE_NOT_APPLICABLE",
                stage="minimum_spanning_tree_computation",
                message=str(exc),
                hint=(
                    "Check the finite simple weighted-graph contract and exact "
                    "rational edge weights."
                ),
            )
        )


MINIMUM_SPANNING_TREE_CAPABILITY: ComputedOperation[
    GraphMinimumSpanningTreeRequest,
    GraphMinimumSpanningTreeResult,
] = ComputedOperation(
    capability_id="graph.spanning_tree.minimum.compute",
    title="Exact weighted minimum spanning tree",
    description=(
        "Compute one deterministic minimum-total-weight spanning tree of a bounded "
        "labelled simple graph over exact rational edge weights. Return the selected "
        "weighted edges, exact total, connected components, and all fundamental-cycle "
        "non-improvement checks; disconnected or empty inputs return a complete "
        "NO_SPANNING_TREE outcome."
    ),
    request_model=GraphMinimumSpanningTreeRequest,
    result_model=GraphMinimumSpanningTreeResult,
    implementation=_execute,
    relation_id="graph.spanning_tree.minimum.relation",
    tags=(
        "graph",
        "weighted-graph",
        "minimum-spanning-tree",
        "mst",
        "spanning-tree",
        "minimum-weight",
        "exact-rational",
        "cycle-property",
    ),
    invalid_request=_INVALID_REQUEST,
    invocation_examples=(
        CapabilityInvocationExample(
            name="four_vertex_weighted_graph",
            description=(
                "Compute an exact minimum spanning tree and its cycle checks."
            ),
            mode=CapabilityMode.EXPLORE,
            input={
                "graph": {
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [
                        {
                            "endpoints": ["a", "b"],
                            "weight": {"num": "1", "den": "1"},
                        },
                        {
                            "endpoints": ["b", "c"],
                            "weight": {"num": "2", "den": "1"},
                        },
                        {
                            "endpoints": ["c", "d"],
                            "weight": {"num": "3", "den": "1"},
                        },
                        {
                            "endpoints": ["a", "d"],
                            "weight": {"num": "5", "den": "1"},
                        },
                        {
                            "endpoints": ["a", "c"],
                            "weight": {"num": "7", "den": "2"},
                        },
                    ],
                }
            },
        ),
    ),
    version="4",
)


__all__ = [
    "MINIMUM_SPANNING_TREE_CAPABILITY",
    "compute_minimum_spanning_tree",
]
