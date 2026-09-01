"""Finite hypergraph operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    CliqueExpansionRequest,
    CliqueExpansionResult,
    DualRequest,
    DualResult,
    EdgeIntersectionGraphRequest,
    EdgeIntersectionGraphResult,
    EdgeIntersectionsRequest,
    EdgeIntersectionsResult,
    HypergraphIndependenceRequest,
    HypergraphIndependenceResult,
    IncidenceGraphRequest,
    IncidenceGraphResult,
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
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.operations import (
    clique_expansion,
    dual,
    edge_intersection_graph,
    edge_intersections,
    incidence_graph,
    independence_number,
    induced_type_profile,
    maximum_edge_matching,
    minimum_transversal,
    parameters,
    vertex_degrees,
)


def _compute_independence_number(
    request: HypergraphIndependenceRequest,
) -> HypergraphIndependenceResult:
    return independence_number(request.hypergraph, request.resource_budget)


def _compute_parameters(request: ParametersRequest) -> ParametersResult:
    return parameters(request.hypergraph)


def _compute_vertex_degrees(request: VertexDegreesRequest) -> VertexDegreesResult:
    return vertex_degrees(request.hypergraph)


def _compute_edge_intersections(
    request: EdgeIntersectionsRequest,
) -> EdgeIntersectionsResult:
    return edge_intersections(request.hypergraph)


def _compute_edge_intersection_graph(
    request: EdgeIntersectionGraphRequest,
) -> EdgeIntersectionGraphResult:
    return edge_intersection_graph(request.hypergraph)


def _compute_dual(request: DualRequest) -> DualResult:
    return dual(request.hypergraph)


def _compute_incidence_graph(
    request: IncidenceGraphRequest,
) -> IncidenceGraphResult:
    return incidence_graph(request.hypergraph)


def _compute_clique_expansion(
    request: CliqueExpansionRequest,
) -> CliqueExpansionResult:
    return clique_expansion(request.hypergraph)


def _compute_induced_type_profile(
    request: InducedTypeProfileRequest,
) -> InducedTypeProfileResult:
    return induced_type_profile(request.hypergraph, request.subset_size)


def _compute_minimum_transversal(
    request: MinimumTransversalRequest,
) -> MinimumTransversalResult:
    return minimum_transversal(request.hypergraph)


def _compute_maximum_edge_matching(
    request: MaximumEdgeMatchingRequest,
) -> MaximumEdgeMatchingResult:
    return maximum_edge_matching(request.hypergraph)


# The Fano-plane-like hypergraph: four vertices and three hyperedges.
_HYPERGRAPH = {
    "vertices": ["a", "b", "c", "d"],
    "edges": [
        ["e1", ["a", "b", "c"]],
        ["e2", ["b", "c", "d"]],
        ["e3", ["a", "d"]],
    ],
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="hypergraph.independence_number.compute",
        title="Compute the independence number of a finite hypergraph",
        description="Compute a maximum vertex subset containing no complete hyperedge. "
        "Return either the exact independence number and a maximizing witness, "
        "or a source-bound feasible incumbent with sound lower and upper bounds "
        "when the bounded exact threshold search does not finish.",
        request_type=HypergraphIndependenceRequest,
        result_type=HypergraphIndependenceResult,
        run=_compute_independence_number,
        tags=(
            "combinatorics",
            "hypergraph",
            "independent-set",
            "optimization",
            "exact-or-unknown",
        ),
        examples=(
            OperationExample(
                name="one_forbidden_triple",
                description="Compute the independence number of one forbidden triple; "
                "every indexed hyperedge must be nonempty.",
                input={
                    "hypergraph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["triple", ["a", "b", "c"]]],
                    },
                    "resource_budget": {
                        "wall_seconds": 5,
                        "max_solver_calls": 16,
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="hypergraph.parameters.compute",
        title="Compute the basic parameters of a finite hypergraph",
        description="Compute the vertex count, edge count, rank, corank, uniform "
        "size, and total incidences of a finite hypergraph.",
        request_type=ParametersRequest,
        result_type=ParametersResult,
        run=_compute_parameters,
        tags=("combinatorics", "hypergraph", "exact"),
        examples=(
            OperationExample(
                name="parameters_of_4_vertex_hypergraph",
                description="Compute the parameters of a 4-vertex, 3-edge hypergraph.",
                input={"hypergraph": _HYPERGRAPH},
            ),
        ),
    ),
    MathTool(
        operation_id="hypergraph.vertex_degrees.compute",
        title="Compute the vertex degrees of a finite hypergraph",
        description="Compute the degree of each vertex and a degree histogram of a "
        "finite hypergraph.",
        request_type=VertexDegreesRequest,
        result_type=VertexDegreesResult,
        run=_compute_vertex_degrees,
        tags=("combinatorics", "hypergraph", "exact"),
        examples=(
            OperationExample(
                name="vertex_degrees_of_4_vertex_hypergraph",
                description="Compute the vertex-degree map of a 4-vertex hypergraph.",
                input={"hypergraph": _HYPERGRAPH},
            ),
        ),
    ),
    MathTool(
        operation_id="hypergraph.edge_intersections.compute",
        title="Compute indexed hypergraph edge-intersection profiles",
        description="Return the exact intersection of every unordered pair of distinct "
        "indexed edges, the complete size histogram, maximum intersection, "
        "and linearity with the first canonical violating pair.",
        request_type=EdgeIntersectionsRequest,
        result_type=EdgeIntersectionsResult,
        run=_compute_edge_intersections,
        tags=("combinatorics", "hypergraph", "intersection", "linearity", "exact"),
        examples=(
            OperationExample(
                name="edge_intersections_of_4_vertex_hypergraph",
                description="Compute every indexed edge-pair intersection and the linearity "
                "profile of a 4-vertex, 3-edge hypergraph; the complete "
                "worst-case intersection ledger must fit the advertised bounds.",
                input={"hypergraph": _HYPERGRAPH},
            ),
        ),
    ),
    MathTool(
        operation_id="hypergraph.edge_intersection_graph.compute",
        title="Compute the edge-intersection graph of a finite hypergraph",
        description="Compute the canonical simple undirected graph whose vertices are "
        "the hypergraph's edge IDs and in which two vertices are adjacent "
        "if and only if the corresponding hyperedges have nonempty "
        "intersection.  The edge-ID carrier admits at most 256 nonempty "
        "Unicode-NFC labels, each using at most 64 UTF-8 bytes, so the result "
        "composes directly with downstream graph operations.",
        request_type=EdgeIntersectionGraphRequest,
        result_type=EdgeIntersectionGraphResult,
        run=_compute_edge_intersection_graph,
        tags=("combinatorics", "hypergraph", "exact"),
        examples=(
            OperationExample(
                name="edge_intersection_graph_of_4_vertex_hypergraph",
                description="Compute the edge-intersection graph of a 4-vertex, 3-edge hypergraph.",
                input={"hypergraph": _HYPERGRAPH},
            ),
        ),
    ),
    MathTool(
        operation_id="hypergraph.dual.compute",
        title="Compute the dual of a finite hypergraph",
        description="Compute the dual hypergraph, transposing vertices and edges so "
        "that the original edges become vertices and the original "
        "vertices become edges.",
        request_type=DualRequest,
        result_type=DualResult,
        run=_compute_dual,
        tags=("combinatorics", "hypergraph", "exact"),
        examples=(
            OperationExample(
                name="dual_of_4_vertex_hypergraph",
                description="Compute the dual of a 4-vertex, 3-edge hypergraph.",
                input={"hypergraph": _HYPERGRAPH},
            ),
        ),
    ),
    MathTool(
        operation_id="hypergraph.incidence_graph.compute",
        title="Compute the bipartite incidence graph of a hypergraph",
        description="Compute the bipartite incidence graph (Levi graph) of a finite "
        "hypergraph, giving vertex-to-edge and edge-to-vertex incidence.",
        request_type=IncidenceGraphRequest,
        result_type=IncidenceGraphResult,
        run=_compute_incidence_graph,
        tags=("combinatorics", "hypergraph", "exact"),
        examples=(
            OperationExample(
                name="incidence_graph_of_4_vertex_hypergraph",
                description="Compute the Levi graph of a 4-vertex, 3-edge hypergraph.",
                input={"hypergraph": _HYPERGRAPH},
            ),
        ),
    ),
    MathTool(
        operation_id="hypergraph.clique_expansion.compute",
        title="Compute the 2-section (clique expansion) of a hypergraph",
        description="Compute the primal/2-section of a finite hypergraph as a canonical "
        "simple undirected graph: two vertices are adjacent if and only if "
        "they share a hyperedge, with edge endpoints in lexical order.",
        request_type=CliqueExpansionRequest,
        result_type=CliqueExpansionResult,
        run=_compute_clique_expansion,
        tags=("combinatorics", "hypergraph", "exact"),
        examples=(
            OperationExample(
                name="clique_expansion_of_4_vertex_hypergraph",
                description="Compute the 2-section of a 4-vertex, 3-edge hypergraph.",
                input={"hypergraph": _HYPERGRAPH},
            ),
        ),
    ),
    MathTool(
        operation_id="hypergraph.induced_type_profile.compute",
        title="Compute the induced uniform type profile of a finite hypergraph",
        description="For each k-subset of the declared vertices, compute the number of "
        "distinct nonempty edges e \u2229 S that arise when the hypergraph is "
        "restricted to that subset S, in lexicographic vertex order.",
        request_type=InducedTypeProfileRequest,
        result_type=InducedTypeProfileResult,
        run=_compute_induced_type_profile,
        tags=("combinatorics", "hypergraph", "exact"),
        examples=(
            OperationExample(
                name="induced_type_profile_of_4_vertex_hypergraph",
                description="Compute the induced 2-subset type profile of a 4-vertex, "
                "3-edge hypergraph.",
                input={"hypergraph": _HYPERGRAPH, "subset_size": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="hypergraph.minimum_transversal.compute",
        title="Compute an exact minimum transversal of a finite hypergraph",
        description="Compute a minimum-cardinality transversal (hitting set): a vertex "
        "set that intersects every nonempty hyperedge, found by exact bounded "
        "search. Empty hyperedges are not admitted.",
        request_type=MinimumTransversalRequest,
        result_type=MinimumTransversalResult,
        run=_compute_minimum_transversal,
        tags=("combinatorics", "hypergraph", "optimization", "exact"),
        examples=(
            OperationExample(
                name="minimum_transversal_of_4_vertex_hypergraph",
                description="Compute a minimum transversal of a 4-vertex, 3-edge hypergraph; "
                "every hyperedge must be nonempty.",
                input={"hypergraph": _HYPERGRAPH},
            ),
        ),
    ),
    MathTool(
        operation_id="hypergraph.maximum_edge_matching.compute",
        title="Compute an exact maximum edge matching of a finite hypergraph",
        description="Compute a maximum-cardinality set of pairwise-disjoint hyperedges "
        "(matching), found by exact bounded search; an all-empty edge family "
        "is handled by a trivial presolve.",
        request_type=MaximumEdgeMatchingRequest,
        result_type=MaximumEdgeMatchingResult,
        run=_compute_maximum_edge_matching,
        tags=("combinatorics", "hypergraph", "optimization", "exact"),
        examples=(
            OperationExample(
                name="maximum_edge_matching_of_4_vertex_hypergraph",
                description="Compute a maximum edge matching of a 4-vertex, 3-edge hypergraph.",
                input={"hypergraph": _HYPERGRAPH},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
