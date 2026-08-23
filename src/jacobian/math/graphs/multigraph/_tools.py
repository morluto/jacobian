"""Exact finite-multigraph flow and cycle-multicover operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.multigraph._models import (
    CycleMulticoverRequest,
    CycleMulticoverResult,
    EulerianCyclesRequest,
    EulerianCyclesResult,
    MultigraphFlowCheckRequest,
    MultigraphFlowCheckResult,
    MultigraphFlowFindRequest,
    MultigraphFlowFindResult,
)
from jacobian.math.graphs.multigraph._operations import (
    check_cycle_multicover,
    check_multigraph_flow,
    compute_eulerian_cycles,
    find_multigraph_flow,
)


def multigraph_operation[
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


_TRIANGLE_GRAPH = {
    "vertex_count": 3,
    "edges": [
        {"edge_id": "e0", "left": 0, "right": 1},
        {"edge_id": "e1", "left": 1, "right": 2},
        {"edge_id": "e2", "left": 2, "right": 0},
    ],
}

MULTIGRAPH_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    multigraph_operation(
        "graph.multigraph.flow.finite_abelian.check",
        "Check a finite-Abelian flow on a loopless multigraph",
        "Recompute every vertex signed divergence exactly for an oriented"
        " finite-Abelian-group-valued flow on a loopless multigraph. Returns"
        " the per-vertex divergence ledger, zero-valued edges, nowhere-zero"
        " flag, and conservation status. Parallel edges are distinguished by"
        " explicit edge IDs.",
        MultigraphFlowCheckRequest,
        MultigraphFlowCheckResult,
        check_multigraph_flow,
        "graph",
        "multigraph",
        "flow",
        "finite-abelian",
        "exact",
        examples=(
            example(
                "triangle_z3_nowhere_zero",
                "A triangle with a nowhere-zero Z/3Z cyclic flow.",
                {
                    "graph": _TRIANGLE_GRAPH,
                    "group": {"moduli": [3]},
                    "edge_values": [
                        {
                            "edge_id": "e0",
                            "orientation": "left_to_right",
                            "value": [1],
                        },
                        {
                            "edge_id": "e1",
                            "orientation": "left_to_right",
                            "value": [1],
                        },
                        {
                            "edge_id": "e2",
                            "orientation": "left_to_right",
                            "value": [1],
                        },
                    ],
                },
            ),
        ),
    ),
    multigraph_operation(
        "graph.multigraph.flow.finite_abelian.find",
        "Search for a finite-Abelian flow on a loopless multigraph",
        "Bounded exhaustive search for a finite-Abelian-group-valued flow on"
        " a loopless multigraph. Returns FOUND with a checked witness,"
        " EXHAUSTED when the complete declared finite search space was covered"
        " and no flow exists, or UNKNOWN when the resource budget was"
        " exceeded. The search space is the product of per-edge group elements"
        " and orientations.",
        MultigraphFlowFindRequest,
        MultigraphFlowFindResult,
        find_multigraph_flow,
        "graph",
        "multigraph",
        "flow",
        "finite-abelian",
        "bounded-search",
        examples=(
            example(
                "triangle_z3_find",
                "Find a nowhere-zero Z/3Z flow on a triangle.",
                {
                    "graph": _TRIANGLE_GRAPH,
                    "group": {"moduli": [3]},
                    "resource_budget": {
                        "max_states": 1000000,
                        "require_nowhere_zero": True,
                    },
                },
            ),
        ),
    ),
    multigraph_operation(
        "graph.multigraph.eulerian_cycles.compute",
        "Decompose an edge multiset into edge-disjoint Eulerian cycles",
        "Compute a deterministic edge-disjoint cycle decomposition of an"
        " edge subset in a loopless multigraph using NetworkX."
        " Returns explicit cycles as closed vertex/edge-ID sequences and a"
        " complete edge-usage profile. Any induced-degree parity is accepted:"
        " Eulerian subsets are fully decomposed with covers_all=True, while"
        " non-Eulerian subsets return an empty decomposition with"
        " covers_all=False.",
        EulerianCyclesRequest,
        EulerianCyclesResult,
        compute_eulerian_cycles,
        "graph",
        "multigraph",
        "eulerian",
        "cycle-decomposition",
        "exact",
        examples=(
            example(
                "triangle_eulerian",
                "Decompose a triangle into an Eulerian cycle.",
                {"graph": _TRIANGLE_GRAPH},
            ),
            example(
                "path_subset_not_eulerian",
                "A two-edge path has odd induced degrees, so the accepted"
                " request returns the empty decomposition with covers_all=False.",
                {"graph": _TRIANGLE_GRAPH, "edge_subset": ["e0", "e1"]},
            ),
        ),
    ),
    multigraph_operation(
        "graph.multigraph.cycle_multicover.check",
        "Check that a cycle family covers each edge exactly k times",
        "Verify that a submitted family of cycles covers every edge of a"
        " loopless multigraph exactly k times. Each cycle is validated"
        " against graph incidence. Cycles may appear in any ordering,"
        " rotation, or reversal; the operation scores per-edge multiplicity,"
        " not one rendering. Returns per-cycle validity, the edge-multiplicity"
        " profile, missing and overcovered edges, and the exact-k-cover flag.",
        CycleMulticoverRequest,
        CycleMulticoverResult,
        check_cycle_multicover,
        "graph",
        "multigraph",
        "cycle",
        "multicover",
        "exact",
        examples=(
            example(
                "triangle_double_cover",
                "Check a double cover of a triangle (each edge covered twice).",
                {
                    "graph": _TRIANGLE_GRAPH,
                    "cycles": [
                        {
                            "vertices": [0, 1, 2, 0],
                            "edge_ids": ["e0", "e1", "e2"],
                        },
                        {
                            "vertices": [0, 2, 1, 0],
                            "edge_ids": ["e2", "e1", "e0"],
                        },
                    ],
                    "target_multiplicity": 2,
                },
            ),
        ),
    ),
)

TOOLS = MULTIGRAPH_OPERATIONS

__all__ = ["TOOLS"]
