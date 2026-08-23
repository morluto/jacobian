"""Induced graph-pattern count operation declaration."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.patterns._models import (
    InducedVertexSubsetPatternCountRequest,
    InducedVertexSubsetPatternCountResult,
)
from jacobian.math.graphs.patterns._operations import (
    compute_induced_vertex_subset_pattern_count,
)

TOOLS: MathTools = (
    MathTool(
        operation_id="graph.induced_vertex_subset_pattern.count",
        version="1",
        title="Count induced vertex-subset copies of a graph pattern",
        description=(
            "Count host vertex subsets whose induced simple graph is isomorphic "
            "to a supplied pattern, once per subset rather than per embedding. "
            "Admission preflights C(|V(host)|, |V(pattern)|), encoded source and "
            "retained-result bytes, C(|V(pattern)|, 2) direct host-edge probes "
            "and explicit local-graph construction per subset, and worst-case "
            "NetworkX VF2++ partial-map work for every subset. It admits at most "
            "5,000 subsets per pass and 64,000,000 work units "
            "across counting and source-bound result replay; these are "
            "conservative current-backend limits."
        ),
        request_type=InducedVertexSubsetPatternCountRequest,
        result_type=InducedVertexSubsetPatternCountResult,
        run=compute_induced_vertex_subset_pattern_count,
        tags=(
            "graph",
            "induced-subgraph",
            "pattern",
            "count",
            "isomorphism",
            "exact",
            "bounded",
        ),
        examples=(
            example(
                "two_induced_p4_in_p5",
                "Count two P4 subsets in P5. Admission preflights 5 subsets, 6 "
                "direct host-edge probes and one local graph per subset, plus "
                "per-subset VF2++ work across two passes; limits are 5,000 subsets "
                "per pass and 64,000,000 work units.",
                {
                    "host": {
                        "vertices": ["a", "b", "c", "d", "e"],
                        "edges": [["a", "b"], ["b", "c"], ["c", "d"], ["d", "e"]],
                    },
                    "pattern": {
                        "vertices": ["w", "x", "y", "z"],
                        "edges": [["w", "x"], ["x", "y"], ["y", "z"]],
                    },
                },
            ),
            example(
                "c4_does_not_induce_p4",
                "Count zero P4 subsets in C4 because its closing edge is retained; both inputs must be canonical finite simple undirected graphs.",
                {
                    "host": {
                        "vertices": ["a", "b", "c", "d"],
                        "edges": [["a", "b"], ["a", "d"], ["b", "c"], ["c", "d"]],
                    },
                    "pattern": {
                        "vertices": ["w", "x", "y", "z"],
                        "edges": [["w", "x"], ["x", "y"], ["y", "z"]],
                    },
                },
            ),
            example(
                "one_induced_two_edge_matching",
                "Count the single full vertex subset of 2K2 inducing 2K2; each subset contributes once regardless of pattern automorphisms.",
                {
                    "host": {
                        "vertices": ["a", "b", "c", "d"],
                        "edges": [["a", "b"], ["c", "d"]],
                    },
                    "pattern": {
                        "vertices": ["w", "x", "y", "z"],
                        "edges": [["w", "x"], ["y", "z"]],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
