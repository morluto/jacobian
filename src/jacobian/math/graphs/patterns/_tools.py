"""Induced graph-pattern count operation declaration."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.patterns._models import (
    MAX_INDUCED_PATTERN_CANDIDATES,
    MAX_INDUCED_PATTERN_TOTAL_WORK_UNITS,
    InducedVertexSubsetPatternCountRequest,
    InducedVertexSubsetPatternCountResult,
)
from jacobian.math.graphs.patterns.operations import (
    induced_vertex_subset_pattern_count,
)


def _run_count(
    request: InducedVertexSubsetPatternCountRequest,
) -> InducedVertexSubsetPatternCountResult:
    return InducedVertexSubsetPatternCountResult._from_kernel(
        host=request.host,
        pattern=request.pattern,
        occurrence_count=induced_vertex_subset_pattern_count(
            request.host, request.pattern
        ),
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.induced_vertex_subset_pattern.count",
        title="Count induced vertex-subset copies of a graph pattern",
        description=(
            "Count host vertex subsets whose induced simple graph is isomorphic "
            "to a pattern, once per subset, not per embedding. Admission preflights "
            "C(|V(host)|, |V(pattern)|), "
            "C(|V(pattern)|, 2) direct host-edge probes, explicit local-graph "
            "construction, and a worst-case NetworkX VF2++ partial-map bound. "
            f"Limits are {MAX_INDUCED_PATTERN_CANDIDATES:,} subsets and {MAX_INDUCED_PATTERN_TOTAL_WORK_UNITS:,} work units; this is a conservative backend envelope."
        ),
        request_type=InducedVertexSubsetPatternCountRequest,
        result_type=InducedVertexSubsetPatternCountResult,
        run=_run_count,
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
            OperationExample(
                name="two_induced_p4_in_p5",
                description="Count two P4 subsets in P5. Admission preflights 5 subsets, 6 "
                "direct host-edge probes and one local graph per subset, plus "
                f"per-subset VF2++ work; limits are {MAX_INDUCED_PATTERN_CANDIDATES:,} subsets "
                f"and {MAX_INDUCED_PATTERN_TOTAL_WORK_UNITS:,} work units.",
                input={
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
            OperationExample(
                name="c4_does_not_induce_p4",
                description="Count zero P4 subsets in C4 because its closing edge is retained; both inputs must be canonical finite simple undirected graphs.",
                input={
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
            OperationExample(
                name="one_induced_two_edge_matching",
                description="Count the single full vertex subset of 2K2 inducing 2K2; each subset contributes once regardless of pattern automorphisms.",
                input={
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
