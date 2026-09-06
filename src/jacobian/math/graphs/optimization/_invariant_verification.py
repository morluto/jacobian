"""Explicit checking of graph-relative invariant claims."""

from jacobian.math.graphs.optimization._chromatic_kernel import build_simple_graph
from jacobian.math.graphs.optimization._invariant_models import (
    GraphCoreRequest,
    GraphCoreResult,
    GraphDiameterResult,
    GraphEdgeConnectivityResult,
    GraphEulerianResult,
    GraphGirthResult,
    GraphRadiusResult,
    GraphSpanningTreeCountResult,
    GraphTriangleCountResult,
    GraphVertexConnectivityResult,
)

type GraphInvariantClaim = (
    GraphCoreResult
    | GraphDiameterResult
    | GraphEdgeConnectivityResult
    | GraphEulerianResult
    | GraphGirthResult
    | GraphRadiusResult
    | GraphSpanningTreeCountResult
    | GraphTriangleCountResult
    | GraphVertexConnectivityResult
)


def verify_graph_invariant(claim: GraphInvariantClaim) -> bool:
    """Check the retained graph's claimed invariant within its 256-vertex domain.

    Only the selected invariant is computed. Diagnostic prose is not part of
    the mathematical relation, and parsing never invokes this checker.
    """
    from jacobian.math.graphs.optimization import _invariants as kernels

    graph = build_simple_graph(claim.graph)
    expected: GraphInvariantClaim
    if isinstance(claim, GraphCoreResult):
        expected = kernels._k_core_execute(
            GraphCoreRequest(graph=claim.graph, k=claim.k)
        )
    elif isinstance(claim, GraphDiameterResult):
        expected = kernels._diameter(graph, claim.graph)
    elif isinstance(claim, GraphEdgeConnectivityResult):
        expected = kernels._edge_connectivity(graph, claim.graph)
    elif isinstance(claim, GraphEulerianResult):
        expected = kernels._eulerian(graph, claim.graph)
    elif isinstance(claim, GraphGirthResult):
        expected = kernels._girth(graph, claim.graph)
    elif isinstance(claim, GraphRadiusResult):
        expected = kernels._radius(graph, claim.graph)
    elif isinstance(claim, GraphSpanningTreeCountResult):
        expected = kernels._spanning_tree_count(graph, claim.graph)
    elif isinstance(claim, GraphTriangleCountResult):
        expected = kernels._triangle_count(graph, claim.graph)
    else:
        expected = kernels._vertex_connectivity(graph, claim.graph)
    return expected.model_dump(exclude={"detail"}) == claim.model_dump(
        exclude={"detail"}
    )
