"""Finite hypergraph operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.hypergraphs._models import (
    CliqueExpansionRequest,
    CliqueExpansionResult,
    DualRequest,
    DualResult,
    EdgeIntersectionsRequest,
    EdgeIntersectionsResult,
    IncidenceGraphRequest,
    IncidenceGraphResult,
    ParametersRequest,
    ParametersResult,
    VertexDegreesRequest,
    VertexDegreesResult,
)
from jacobian.math.hypergraphs._operations import (
    compute_clique_expansion,
    compute_dual,
    compute_edge_intersections,
    compute_incidence_graph,
    compute_parameters,
    compute_vertex_degrees,
)


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version="1",
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


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
    _op(
        "hypergraph.parameters.compute",
        "Compute the basic parameters of a finite hypergraph",
        "Compute the vertex count, edge count, rank, corank, uniform "
        "size, and total incidences of a finite hypergraph.",
        ParametersRequest,
        ParametersResult,
        compute_parameters,
        "combinatorics",
        "hypergraph",
        "exact",
        examples=(
            example(
                "parameters_of_4_vertex_hypergraph",
                "Compute the parameters of a 4-vertex, 3-edge hypergraph.",
                {"hypergraph": _HYPERGRAPH},
            ),
        ),
    ),
    _op(
        "hypergraph.vertex_degrees.compute",
        "Compute the vertex degrees of a finite hypergraph",
        "Compute the degree of each vertex and a degree histogram of a "
        "finite hypergraph.",
        VertexDegreesRequest,
        VertexDegreesResult,
        compute_vertex_degrees,
        "combinatorics",
        "hypergraph",
        "exact",
        examples=(
            example(
                "vertex_degrees_of_4_vertex_hypergraph",
                "Compute the vertex-degree map of a 4-vertex hypergraph.",
                {"hypergraph": _HYPERGRAPH},
            ),
        ),
    ),
    _op(
        "hypergraph.edge_intersections.compute",
        "Compute indexed hypergraph edge-intersection profiles",
        "Return the exact intersection of every unordered pair of distinct "
        "indexed edges, the complete size histogram, maximum intersection, "
        "and linearity with the first canonical violating pair.",
        EdgeIntersectionsRequest,
        EdgeIntersectionsResult,
        compute_edge_intersections,
        "combinatorics",
        "hypergraph",
        "intersection",
        "linearity",
        "exact",
        examples=(
            example(
                "edge_intersections_of_4_vertex_hypergraph",
                "Compute every indexed edge-pair intersection and the linearity "
                "profile of a 4-vertex, 3-edge hypergraph; the complete "
                "worst-case intersection ledger must fit the advertised bounds.",
                {"hypergraph": _HYPERGRAPH},
            ),
        ),
    ),
    _op(
        "hypergraph.dual.compute",
        "Compute the dual of a finite hypergraph",
        "Compute the dual hypergraph, transposing vertices and edges so "
        "that the original edges become vertices and the original "
        "vertices become edges.",
        DualRequest,
        DualResult,
        compute_dual,
        "combinatorics",
        "hypergraph",
        "exact",
        examples=(
            example(
                "dual_of_4_vertex_hypergraph",
                "Compute the dual of a 4-vertex, 3-edge hypergraph.",
                {"hypergraph": _HYPERGRAPH},
            ),
        ),
    ),
    _op(
        "hypergraph.incidence_graph.compute",
        "Compute the bipartite incidence graph of a hypergraph",
        "Compute the bipartite incidence graph (Levi graph) of a finite "
        "hypergraph, giving vertex-to-edge and edge-to-vertex incidence.",
        IncidenceGraphRequest,
        IncidenceGraphResult,
        compute_incidence_graph,
        "combinatorics",
        "hypergraph",
        "exact",
        examples=(
            example(
                "incidence_graph_of_4_vertex_hypergraph",
                "Compute the Levi graph of a 4-vertex, 3-edge hypergraph.",
                {"hypergraph": _HYPERGRAPH},
            ),
        ),
    ),
    _op(
        "hypergraph.clique_expansion.compute",
        "Compute the 2-section (clique expansion) of a hypergraph",
        "Compute the primal/2-section of a finite hypergraph as a canonical "
        "simple undirected graph: two vertices are adjacent if and only if "
        "they share a hyperedge, with edge endpoints in lexical order.",
        CliqueExpansionRequest,
        CliqueExpansionResult,
        compute_clique_expansion,
        "combinatorics",
        "hypergraph",
        "exact",
        examples=(
            example(
                "clique_expansion_of_4_vertex_hypergraph",
                "Compute the 2-section of a 4-vertex, 3-edge hypergraph.",
                {"hypergraph": _HYPERGRAPH},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
