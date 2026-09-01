"""Graph morphism operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.morphisms._models import (
    FixedLengthCycleRequest,
    FixedLengthCycleResult,
    HomomorphismCheckRequest,
    HomomorphismCheckResult,
    SubgraphPatternFindRequest,
    SubgraphPatternFindResult,
)
from jacobian.math.graphs.morphisms.operations import (
    fixed_length_cycle,
    homomorphism_check,
    subgraph_pattern_find,
)


def _compute_homomorphism_check(
    request: HomomorphismCheckRequest,
) -> HomomorphismCheckResult:
    return homomorphism_check(request.vertex_map)


def _compute_fixed_length_cycle(
    request: FixedLengthCycleRequest,
) -> FixedLengthCycleResult:
    return fixed_length_cycle(request.graph, request.length)


def _compute_subgraph_pattern_find(
    request: SubgraphPatternFindRequest,
) -> SubgraphPatternFindResult:
    return subgraph_pattern_find(request.pattern, request.host)


CYCLE_C4_WITH_CHORD = {
    "graph": {
        "vertices": ["a", "b", "c", "d"],
        "edges": [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["c", "d"]],
    },
    "length": 3,
}
# For the chorded case we need at least triangle a-b-c. Using edges a-b, b-c, a-c plus rest.
CYCLE_C4_WITH_CHORD_SIMPLE = {
    "graph": {
        "vertices": ["a", "b", "c", "d"],
        "edges": [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["c", "d"]],
    },
    "length": 3,
}
CYCLE_C4_PLAIN = {
    "graph": {
        "vertices": ["a", "b", "c", "d"],
        "edges": [["a", "b"], ["a", "d"], ["b", "c"], ["c", "d"]],
    },
    "length": 3,
}
SUBGRAPH_TRIANGLE_IN_C4_CHORD = {
    "pattern": {
        "vertices": ["x", "y", "z"],
        "edges": [["x", "y"], ["x", "z"], ["y", "z"]],
    },
    "host": {
        "vertices": ["a", "b", "c", "d"],
        "edges": [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["c", "d"]],
    },
}
SUBGRAPH_P3_NOT_IN_MATCHING = {
    "pattern": {
        "vertices": ["x", "y", "z"],
        "edges": [["x", "y"], ["y", "z"]],
    },
    "host": {
        "vertices": ["a", "b", "c", "d"],
        "edges": [["a", "b"], ["c", "d"]],
    },
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.homomorphism.check",
        title="Check one complete graph vertex map",
        description="Check a complete canonical vertex map between two labelled simple "
        "graphs. Returns either the source-bound checked homomorphism or the "
        "first source edge whose image is not a target edge.",
        request_type=HomomorphismCheckRequest,
        result_type=HomomorphismCheckResult,
        run=_compute_homomorphism_check,
        tags=("graph", "homomorphism", "exact"),
        examples=(
            OperationExample(
                name="identity_homomorphism",
                description="Check the canonical identity map on a single edge graph.",
                input={
                    "vertex_map": {
                        "source_graph": {
                            "vertices": ["a", "b"],
                            "edges": [["a", "b"]],
                        },
                        "target_graph": {
                            "vertices": ["a", "b"],
                            "edges": [["a", "b"]],
                        },
                        "rows": [
                            {"source_vertex": "a", "target_vertex": "a"},
                            {"source_vertex": "b", "target_vertex": "b"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.cycle.fixed_length.decide",
        title="Decide a fixed-length simple cycle",
        description="Given a bounded simple graph and an integer k >= 3, decide whether "
        "the graph contains a simple cycle of exactly length k, returning an "
        "ordered cycle witness when one exists. The cycle is a subgraph and "
        "may have chords; this is distinct from girth (shortest cycle) and "
        "from Hamiltonicity (spanning). Accepts the canonical "
        "`SimpleUndirectedGraph` so `explicit_graph` output composes directly.",
        request_type=FixedLengthCycleRequest,
        result_type=FixedLengthCycleResult,
        run=_compute_fixed_length_cycle,
        tags=("graph", "cycle", "subgraph"),
        examples=(
            OperationExample(
                name="c4_with_chord_has_triangle",
                description=(
                    "A 4-cycle with a chord contains a 3-cycle (triangle); "
                    "length k is 3..vertex count. Preconditions: at most 64 "
                    "vertices and inside the path-search budget."
                ),
                input=CYCLE_C4_WITH_CHORD_SIMPLE,
            ),
            OperationExample(
                name="c4_plain_no_triangle",
                description=(
                    "A plain 4-cycle has no 3-cycle. Preconditions: length 3..64 "
                    "and at most the vertex count, and the per-pass path budget holds."
                ),
                input=CYCLE_C4_PLAIN,
            ),
        ),
    ),
    MathTool(
        operation_id="graph.subgraph_pattern.find",
        title="Find a subgraph-pattern embedding",
        description="Given bounded canonical simple graphs pattern H and host G, find an "
        "injective non-induced embedding. Returns one vertex map in pattern order "
        "when found. Assignment search is admission-bounded; runtime candidate-"
        "check exhaustion returns BUDGET_EXCEEDED, a typed non-conclusion. Both "
        "returned maps are bounded by the admitted pattern cardinality.",
        request_type=SubgraphPatternFindRequest,
        result_type=SubgraphPatternFindResult,
        run=_compute_subgraph_pattern_find,
        tags=("graph", "subgraph", "monomorphism"),
        examples=(
            OperationExample(
                name="triangle_in_c4_with_chord",
                description=(
                    "A triangle pattern embeds in a 4-cycle-with-chord host. "
                    "Preconditions: pattern at most 64 vertices, no larger than "
                    "host and inside the assignment budget."
                ),
                input=SUBGRAPH_TRIANGLE_IN_C4_CHORD,
            ),
            OperationExample(
                name="p3_not_in_matching",
                description=(
                    "A path P3 does not embed in two disjoint host edges. "
                    "Preconditions: at most 64 pattern vertices, no larger than "
                    "the host, and the per-pass budget holds."
                ),
                input=SUBGRAPH_P3_NOT_IN_MATCHING,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
