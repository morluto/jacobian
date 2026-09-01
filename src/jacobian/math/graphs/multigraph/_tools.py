"""Exact finite-multigraph flow and cycle-multicover operation declarations."""

from typing import Any

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
from jacobian.math.graphs.multigraph.operations import (
    cycle_multicover,
    eulerian_cycles,
    multigraph_flow_check,
    multigraph_flow_find,
)


def _compute_multigraph_flow_check(
    request: MultigraphFlowCheckRequest,
) -> MultigraphFlowCheckResult:
    return multigraph_flow_check(request.graph, request.group, request.edge_values)


def _compute_multigraph_flow_find(
    request: MultigraphFlowFindRequest,
) -> MultigraphFlowFindResult:
    return multigraph_flow_find(request.graph, request.group, request.resource_budget)


def _compute_eulerian_cycles(
    request: EulerianCyclesRequest,
) -> EulerianCyclesResult:
    return eulerian_cycles(request.graph, request.edge_subset)


def _compute_cycle_multicover(
    request: CycleMulticoverRequest,
) -> CycleMulticoverResult:
    return cycle_multicover(request.graph, request.cycles, request.target_multiplicity)


_TRIANGLE_GRAPH = {
    "vertex_count": 3,
    "edges": [
        {"edge_id": "e0", "left": 0, "right": 1},
        {"edge_id": "e1", "left": 1, "right": 2},
        {"edge_id": "e2", "left": 2, "right": 0},
    ],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.multigraph.flow.finite_abelian.check",
        title="Check a finite-Abelian flow on a loopless multigraph",
        description="Recompute every vertex signed divergence exactly for an oriented"
        " finite-Abelian-group-valued flow on a loopless multigraph. Returns"
        " the per-vertex divergence ledger, zero-valued edges, nowhere-zero"
        " flag, and conservation status. Parallel edges are distinguished by"
        " explicit edge IDs.",
        request_type=MultigraphFlowCheckRequest,
        result_type=MultigraphFlowCheckResult,
        run=_compute_multigraph_flow_check,
        tags=("graph", "multigraph", "flow", "finite-abelian", "exact"),
        examples=(
            OperationExample(
                name="triangle_z3_nowhere_zero",
                description="A triangle with a nowhere-zero Z/3Z cyclic flow.",
                input={
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
    MathTool(
        operation_id="graph.multigraph.flow.finite_abelian.find",
        title="Search for a finite-Abelian flow on a loopless multigraph",
        description="Bounded exhaustive search for a finite-Abelian-group-valued flow on"
        " a loopless multigraph. Returns FOUND with a checked witness,"
        " EXHAUSTED when the complete declared finite search space was covered"
        " and no flow exists, or UNKNOWN when the resource budget was"
        " exceeded. The search space is the product of per-edge group elements"
        " and orientations.",
        request_type=MultigraphFlowFindRequest,
        result_type=MultigraphFlowFindResult,
        run=_compute_multigraph_flow_find,
        tags=("graph", "multigraph", "flow", "finite-abelian", "bounded-search"),
        examples=(
            OperationExample(
                name="triangle_z3_find",
                description="Find a nowhere-zero Z/3Z flow on a triangle.",
                input={
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
    MathTool(
        operation_id="graph.multigraph.eulerian_cycles.compute",
        title="Decompose an edge multiset into edge-disjoint Eulerian cycles",
        description="Compute a deterministic edge-disjoint cycle decomposition of an"
        " edge subset in a loopless multigraph using NetworkX."
        " Returns explicit cycles as closed vertex/edge-ID sequences and a"
        " complete edge-usage profile. Any induced-degree parity is accepted:"
        " Eulerian subsets are fully decomposed with covers_all=True, while"
        " non-Eulerian subsets return an empty decomposition with"
        " covers_all=False.",
        request_type=EulerianCyclesRequest,
        result_type=EulerianCyclesResult,
        run=_compute_eulerian_cycles,
        tags=("graph", "multigraph", "eulerian", "cycle-decomposition", "exact"),
        examples=(
            OperationExample(
                name="triangle_eulerian",
                description="Decompose a triangle into an Eulerian cycle.",
                input={"graph": _TRIANGLE_GRAPH},
            ),
            OperationExample(
                name="path_subset_not_eulerian",
                description="A two-edge path has odd induced degrees, so the accepted"
                " request returns the empty decomposition with covers_all=False.",
                input={"graph": _TRIANGLE_GRAPH, "edge_subset": ["e0", "e1"]},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.multigraph.cycle_multicover.check",
        title="Check that a cycle family covers each edge exactly k times",
        description="Verify that a submitted family of cycles covers every edge of a"
        " loopless multigraph exactly k times. Each cycle is validated"
        " against graph incidence. Cycles may appear in any ordering,"
        " rotation, or reversal; the operation scores per-edge multiplicity,"
        " not one rendering. Returns per-cycle validity, the edge-multiplicity"
        " profile, missing and overcovered edges, and the exact-k-cover flag.",
        request_type=CycleMulticoverRequest,
        result_type=CycleMulticoverResult,
        run=_compute_cycle_multicover,
        tags=("graph", "multigraph", "cycle", "multicover", "exact"),
        examples=(
            OperationExample(
                name="triangle_double_cover",
                description="Check a double cover of a triangle (each edge covered twice).",
                input={
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


__all__ = ["TOOLS"]
