"""Typed declarations for graph cycle and subgraph-pattern operations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.cycle_pattern._models import (
    FixedLengthCycleRequest,
    FixedLengthCycleResult,
    SubgraphPatternRequest,
    SubgraphPatternResult,
)
from jacobian.math.graphs.cycle_pattern._operations import (
    decide_fixed_length_cycle,
    find_subgraph_pattern,
)


def cycle_pattern_operation[
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


def _named_graph(names: list[str], edges: list[list[str]]) -> dict[str, Any]:
    return {
        "graph_schema_version": "1",
        "vertices": names,
        "edges": edges,
    }


_CYCLE_DECIDE_EXAMPLE: dict[str, Any] = {
    "graph": _named_graph(
        ["a", "b", "c", "d"],
        [["a", "b"], ["b", "c"], ["c", "d"], ["a", "d"]],
    ),
    "length": 4,
}

_NO_CYCLE_EXAMPLE: dict[str, Any] = {
    "graph": _named_graph(
        ["a", "b", "c", "d"],
        [["a", "b"], ["b", "c"], ["c", "d"]],
    ),
    "length": 4,
}

_TRIANGLE_EXAMPLE: dict[str, Any] = {
    "graph": _named_graph(
        ["a", "b", "c"],
        [["a", "b"], ["b", "c"], ["a", "c"]],
    ),
    "length": 3,
}

_SUBGRAPH_EXAMPLE: dict[str, Any] = {
    "host": _named_graph(
        ["a", "b", "c", "d", "e"],
        [["a", "b"], ["b", "c"], ["c", "d"], ["d", "e"], ["a", "e"], ["a", "c"]],
    ),
    "pattern": _named_graph(["a", "b", "c"], [["a", "b"], ["b", "c"]]),
}

_NO_SUBGRAPH_EXAMPLE: dict[str, Any] = {
    "host": _named_graph(["a", "b", "c"], [["a", "b"]]),
    "pattern": _named_graph(
        ["a", "b", "c"],
        [["a", "b"], ["b", "c"], ["a", "c"]],
    ),
}


CYCLE_PATTERN_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    cycle_pattern_operation(
        "graph.cycle.fixed_length.decide",
        "Decide whether a graph contains a simple cycle of a given length",
        "Given a bounded simple undirected graph G and an integer k, decide "
        "whether G contains a simple cycle of exactly length k. Returns an "
        "explicit ordered cycle witness when one exists, or a complete "
        "DOES_NOT_EXIST result after exhaustive bounded search. If the "
        "two-million-node alternating-path search budget is exhausted "
        "first, the result is instead the undecided SEARCH_BUDGET_EXCEEDED. "
        "Distinct from girth, Hamiltonian cycles, and induced cycles: the "
        "requested cycle may have chords.",
        FixedLengthCycleRequest,
        FixedLengthCycleResult,
        decide_fixed_length_cycle,
        "graph",
        "cycle",
        "exact",
        examples=(
            example(
                "four_cycle",
                "A 4-cycle on 4 vertices with 4 edges has a simple cycle of length 4.",
                _CYCLE_DECIDE_EXAMPLE,
            ),
            example(
                "no_cycle_path",
                "A path graph on 4 vertices has no 4-cycle.",
                _NO_CYCLE_EXAMPLE,
            ),
            example(
                "triangle",
                "A triangle graph has a cycle of length 3.",
                _TRIANGLE_EXAMPLE,
            ),
        ),
    ),
    cycle_pattern_operation(
        "graph.subgraph.pattern.find",
        "Find a subgraph embedding of a pattern graph into a host graph",
        "Given bounded simple undirected graphs G (host) and H (pattern), "
        "find an injective embedding of H into G that preserves all edges, "
        "or establish after a bounded backtracking search (degree pruning, "
        "deterministic recursion-node budget) that no such embedding exists. "
        "When the budget is exhausted before deciding, the result says so "
        "instead of claiming completeness. Returns the mapping from pattern "
        "vertices to host vertices as a witness validated against the "
        "source graphs.",
        SubgraphPatternRequest,
        SubgraphPatternResult,
        find_subgraph_pattern,
        "graph",
        "subgraph",
        "pattern",
        "exact",
        examples=(
            example(
                "path_in_pentagon",
                "A path P3 embeds into a pentagon with a chord.",
                _SUBGRAPH_EXAMPLE,
            ),
            example(
                "triangle_not_in_edge",
                "A triangle cannot embed into a single-edge graph.",
                _NO_SUBGRAPH_EXAMPLE,
            ),
        ),
    ),
)

TOOLS = CYCLE_PATTERN_OPERATIONS

__all__ = ["TOOLS"]
