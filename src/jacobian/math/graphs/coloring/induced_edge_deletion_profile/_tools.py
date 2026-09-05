"""Induced edge deletion profile operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.coloring.induced_edge_deletion_profile._models import (
    InducedEdgeDeletionProfileRequest,
    InducedEdgeDeletionProfileResult,
)
from jacobian.math.graphs.coloring.induced_edge_deletion_profile.operations import (
    compute_induced_edge_deletion_profile,
)


def compute_induced_edge_deletion_profile_op(
    request: InducedEdgeDeletionProfileRequest,
) -> InducedEdgeDeletionProfileResult:
    return compute_induced_edge_deletion_profile(
        request.graph, request.r, request.solver_conflicts
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.coloring.induced_edge_deletion_profile.compute",
        title="Profile induced subgraphs by minimum edge deletion to r-colourability",
        description=(
            "For a finite labelled simple graph G and integer r>=1, return the "
            "induced profile D_{G,r}(S)=min{|F|: F subset E(G[S]) and chi(G[S]-F) <= r} "
            "for every vertex subset S, with one canonical lexicographically smallest "
            "attaining edge set F per S. The result is exact over all 2^n "
            "vertex subsets; the derived per-size maximum max_{|S|=m} D(S) is exposed "
            "as a deterministic projection. The graph is limited to at most 8 vertices (256 rows) "
            "and the aggregate Z3 conflict ledger is bounded."
        ),
        request_type=InducedEdgeDeletionProfileRequest,
        result_type=InducedEdgeDeletionProfileResult,
        run=compute_induced_edge_deletion_profile_op,
        tags=("graph", "coloring", "induced", "edge-deletion", "profile", "exact"),
        examples=(
            OperationExample(
                name="triangle_r2_profile",
                description=(
                    "K3 at r=2: the whole vertex set needs one source-edge deletion to "
                    "become bipartite, every proper induced subgraph needs zero; the graph "
                    "has at most 8 vertices and r is at least 1."
                ),
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
                    },
                    "r": 2,
                },
            ),
            OperationExample(
                name="path4_r2_already_bipartite",
                description=(
                    "P4 at r=2 is already bipartite on every induced subgraph, so every "
                    "row has min_deletions zero; the request must respect the 8-vertex and "
                    "solver-conflict bounds."
                ),
                input={
                    "graph": {
                        "vertices": ["0", "1", "2", "3"],
                        "edges": [["0", "1"], ["1", "2"], ["2", "3"]],
                    },
                    "r": 2,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
