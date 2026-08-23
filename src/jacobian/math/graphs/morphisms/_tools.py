"""Graph morphism operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.morphisms._models import (
    CoreCheckRequest,
    CoreCheckResult,
    FixedLengthCycleRequest,
    FixedLengthCycleResult,
    HomomorphismCheckRequest,
    HomomorphismCheckResult,
    HomomorphismFindRequest,
    HomomorphismFindResult,
    RetractionCheckRequest,
    RetractionCheckResult,
    SubgraphPatternFindRequest,
    SubgraphPatternFindResult,
)
from jacobian.math.graphs.morphisms._operations import (
    compute_core_check,
    compute_fixed_length_cycle,
    compute_homomorphism_check,
    compute_homomorphism_find,
    compute_retraction_check,
    compute_subgraph_pattern_find,
)


def _op[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


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
    _op(
        "graph.homomorphism.check",
        "Check if a vertex map is a graph homomorphism",
        "Check whether a given vertex map from source to target preserves "
        "all edges of the source graph.",
        HomomorphismCheckRequest,
        HomomorphismCheckResult,
        compute_homomorphism_check,
        "graph",
        "homomorphism",
        "exact",
        examples=(
            example(
                "identity_homomorphism",
                "Check the identity map on a single edge graph.",
                {
                    "source_graph": {
                        "vertex_count": 2,
                        "edges": [[0, 1]],
                    },
                    "target_graph": {
                        "vertex_count": 2,
                        "edges": [[0, 1]],
                    },
                    "vertex_map": [0, 1],
                },
            ),
        ),
    ),
    _op(
        "graph.homomorphism.find",
        "Find a graph homomorphism if one exists",
        "Search for a homomorphism from the source graph to the target graph "
        "using backtracking. Returns whether a homomorphism exists and a "
        "witness vertex map.",
        HomomorphismFindRequest,
        HomomorphismFindResult,
        compute_homomorphism_find,
        "graph",
        "homomorphism",
        "exact",
        examples=(
            example(
                "k2_to_k2",
                "Find a homomorphism from K2 to K2.",
                {
                    "source_graph": {
                        "vertex_count": 2,
                        "edges": [[0, 1]],
                    },
                    "target_graph": {
                        "vertex_count": 2,
                        "edges": [[0, 1]],
                    },
                },
            ),
        ),
    ),
    _op(
        "graph.core.check",
        "Check if a graph is a core",
        "Check whether a graph is a core, i.e., has no non-injective "
        "endomorphism. Returns true if the graph is a core.",
        CoreCheckRequest,
        CoreCheckResult,
        compute_core_check,
        "graph",
        "core",
        "exact",
        examples=(
            example(
                "k2_is_core",
                "Check if K2 is a core.",
                {
                    "graph": {
                        "vertex_count": 2,
                        "edges": [[0, 1]],
                    },
                },
            ),
        ),
    ),
    _op(
        "graph.retraction.check",
        "Check if a retraction onto a subgraph exists",
        "Check whether there exists a homomorphism from the graph to an "
        "subgraph induced by the given vertices that fixes every vertex of "
        "the subgraph.",
        RetractionCheckRequest,
        RetractionCheckResult,
        compute_retraction_check,
        "graph",
        "retraction",
        "exact",
        examples=(
            example(
                "k3_retract_to_k2",
                "Check retraction from K3 to an edge.",
                {
                    "graph": {
                        "vertex_count": 3,
                        "edges": [[0, 1], [1, 2], [0, 2]],
                    },
                    "subgraph_vertices": [0, 1],
                },
            ),
        ),
    ),
    _op(
        "graph.cycle.fixed_length.decide",
        "Decide a fixed-length simple cycle",
        "Given a bounded simple graph and an integer k >= 3, decide whether "
        "the graph contains a simple cycle of exactly length k, returning an "
        "ordered cycle witness when one exists. The cycle is a subgraph and "
        "may have chords; this is distinct from girth (shortest cycle) and "
        "from Hamiltonicity (spanning). Accepts the canonical "
        "`SimpleUndirectedGraph` so `explicit_graph` output composes directly.",
        FixedLengthCycleRequest,
        FixedLengthCycleResult,
        compute_fixed_length_cycle,
        "graph",
        "cycle",
        "subgraph",
        examples=(
            example(
                "c4_with_chord_has_triangle",
                (
                    "A 4-cycle with a chord contains a 3-cycle (triangle); "
                    "length k is 3..vertex count. Preconditions: at most 20 "
                    "vertices, inside the path budget, and enough output "
                    "headroom for the echoed source graph."
                ),
                CYCLE_C4_WITH_CHORD_SIMPLE,
            ),
            example(
                "c4_plain_no_triangle",
                (
                    "A plain 4-cycle has no 3-cycle. Preconditions: length 3..20 "
                    "and at most the vertex count, the per-pass path budget "
                    "holds, and the retained graph plus result envelope fit "
                    "the canonical output limit."
                ),
                CYCLE_C4_PLAIN,
            ),
        ),
    ),
    _op(
        "graph.subgraph_pattern.find",
        "Find a subgraph-pattern embedding",
        "Given bounded canonical simple graphs pattern H and host G, find an "
        "injective non-induced embedding. Returns one vertex map in pattern order "
        "when found. Assignment search is admission-bounded; runtime candidate-"
        "check exhaustion returns BUDGET_EXCEEDED, a typed non-conclusion. Both "
        "graphs and the result envelope must fit the canonical output limit.",
        SubgraphPatternFindRequest,
        SubgraphPatternFindResult,
        compute_subgraph_pattern_find,
        "graph",
        "subgraph",
        "monomorphism",
        examples=(
            example(
                "triangle_in_c4_with_chord",
                (
                    "A triangle pattern embeds in a 4-cycle-with-chord host. "
                    "Preconditions: pattern at most 20 vertices, no larger than "
                    "host, inside the assignment budget, and enough "
                    "output headroom for the echoed sources."
                ),
                SUBGRAPH_TRIANGLE_IN_C4_CHORD,
            ),
            example(
                "p3_not_in_matching",
                (
                    "A path P3 does not embed in two disjoint host edges. "
                    "Preconditions: at most 20 pattern vertices, no larger than "
                    "the host, the per-pass budget holds, and retained graphs "
                    "plus the result envelope fit the canonical output limit."
                ),
                SUBGRAPH_P3_NOT_IN_MATCHING,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
