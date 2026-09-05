"""Clique candidate hypergraph operation declarations."""

from jacobian.catalog.models import (
    MathTool,
    MathTools,
    OperationExample,
)
from jacobian.math.graphs.clique_candidate_hypergraph._models import (
    AllCliqueCandidatesRequest,
    CliqueCandidateHypergraphResult,
)
from jacobian.math.graphs.clique_candidate_hypergraph.operations import (
    construct_all_clique_candidate_hypergraph,
)


def _compute_all_clique_candidates(
    request: AllCliqueCandidatesRequest,
) -> CliqueCandidateHypergraphResult:
    return construct_all_clique_candidate_hypergraph(request.graph)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.clique_candidate_hypergraph.construct",
        title="Construct the all-clique candidate hypergraph of a graph",
        description=(
            "Construct a source-bound FiniteHypergraph whose vertices are "
            "the graph's edges as resources and whose hyperedges are every "
            "nontrivial clique (order at least two, including nonmaximal "
            "ones), each holding exactly its internal-edge resources. "
            "Retains the resource-to-edge and candidate-to-vertex-subset "
            "maps for weighted selection and conflict analysis. These are the "
            "complete candidate supports for integral and fractional "
            "edge-clique partitions, including every clique size."
        ),
        request_type=AllCliqueCandidatesRequest,
        result_type=CliqueCandidateHypergraphResult,
        run=_compute_all_clique_candidates,
        tags=("graph", "hypergraph", "clique", "exact"),
        discovery_terms=(
            "edge clique partition all nontrivial cliques",
            "edge-disjoint complete subgraphs every clique size",
            "integral and fractional edge partition candidates",
        ),
        examples=(
            OperationExample(
                name="bowtie_candidates",
                description=(
                    "Bow-tie graph with maximal cliques abc and ade: six "
                    "edge resources and eight candidates (six edges plus "
                    "two triangles)."
                ),
                input={
                    "graph": {
                        "vertices": ["a", "b", "c", "d", "e"],
                        "edges": [
                            ["a", "b"],
                            ["a", "c"],
                            ["a", "d"],
                            ["a", "e"],
                            ["b", "c"],
                            ["d", "e"],
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
