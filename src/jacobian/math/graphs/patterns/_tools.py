"""Induced graph-pattern count operation declaration."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.patterns._models import (
    MAX_INDUCED_PATTERN_SUBSETS_PER_PASS,
    MAX_INDUCED_PATTERN_TOTAL_WORK_UNITS,
    InducedVertexSubsetPatternCountRequest,
    InducedVertexSubsetPatternCountResult,
)
from jacobian.math.graphs.patterns._operations import (
    compute_induced_vertex_subset_pattern_count,
)

TOOLS: MathTools = (
    MathTool(
        operation_id="graph.induced_vertex_subset_pattern.count",
        title="Count induced vertex-subset copies of a graph pattern",
        description=(
            "Count host vertex subsets whose induced simple graph is isomorphic "
            "to a pattern, once per subset, not per embedding. Admission preflights "
            "C(|V(host)|, |V(pattern)|), exact source/result bytes, "
            "C(|V(pattern)|, 2) direct host-edge probes, explicit local-graph "
            "construction, and a worst-case NetworkX VF2++ partial-map bound. "
            f"Limits are {MAX_INDUCED_PATTERN_SUBSETS_PER_PASS:,} subsets per pass and {MAX_INDUCED_PATTERN_TOTAL_WORK_UNITS:,} work units across "
            "counting and source-bound replay; this is a conservative backend envelope."
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
                f"per-subset VF2++ work across two passes; limits are {MAX_INDUCED_PATTERN_SUBSETS_PER_PASS:,} subsets "
                f"per pass and {MAX_INDUCED_PATTERN_TOTAL_WORK_UNITS:,} work units.",
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
