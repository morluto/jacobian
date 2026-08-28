"""Graph morphism operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def _op[RequestT: StrictModel, ResultT: StrictModel](
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
        "Check one complete graph vertex map",
        "Check a complete canonical vertex map between two labelled simple "
        "graphs. Returns either the source-bound checked homomorphism or the "
        "first source edge whose image is not a target edge.",
        HomomorphismCheckRequest,
        HomomorphismCheckResult,
        _compute_homomorphism_check,
        "graph",
        "homomorphism",
        "exact",
        examples=(
            example(
                "identity_homomorphism",
                "Check the canonical identity map on a single edge graph.",
                {
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
        _compute_fixed_length_cycle,
        "graph",
        "cycle",
        "subgraph",
        examples=(
            example(
                "c4_with_chord_has_triangle",
                (
                    "A 4-cycle with a chord contains a 3-cycle (triangle); "
                    "length k is 3..vertex count. Preconditions: at most 64 "
                    "vertices, inside the path budget, and enough output "
                    "headroom for the echoed source graph."
                ),
                CYCLE_C4_WITH_CHORD_SIMPLE,
            ),
            example(
                "c4_plain_no_triangle",
                (
                    "A plain 4-cycle has no 3-cycle. Preconditions: length 3..64 "
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
        _compute_subgraph_pattern_find,
        "graph",
        "subgraph",
        "monomorphism",
        examples=(
            example(
                "triangle_in_c4_with_chord",
                (
                    "A triangle pattern embeds in a 4-cycle-with-chord host. "
                    "Preconditions: pattern at most 64 vertices, no larger than "
                    "host, inside the assignment budget, and enough "
                    "output headroom for the echoed sources."
                ),
                SUBGRAPH_TRIANGLE_IN_C4_CHORD,
            ),
            example(
                "p3_not_in_matching",
                (
                    "A path P3 does not embed in two disjoint host edges. "
                    "Preconditions: at most 64 pattern vertices, no larger than "
                    "the host, the per-pass budget holds, and retained graphs "
                    "plus the result envelope fit the canonical output limit."
                ),
                SUBGRAPH_P3_NOT_IN_MATCHING,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
