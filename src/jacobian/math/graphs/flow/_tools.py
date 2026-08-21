"""Exact graph flow operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.flow._models import (
    EdgeDisjointPathsRequest,
    EdgeDisjointPathsResult,
    MaxFlowRequest,
    MaxFlowResult,
    MinCostFlowRequest,
    MinCostFlowResult,
    MinCutRequest,
    MinCutResult,
)
from jacobian.math.graphs.flow._operations import (
    compute_edge_disjoint_paths,
    compute_max_flow,
    compute_min_cost_flow,
    compute_min_cut,
)


def graph_flow_operation[
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


GRAPH_FLOW_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    graph_flow_operation(
        "graph.flow.maximum.compute",
        "Compute the maximum flow in a capacitated graph",
        "Compute the maximum flow value between source and sink in a directed capacitated graph using NetworkX. Returns the flow value and a per-edge flow decomposition so the caller can independently verify conservation and capacity constraints.",
        MaxFlowRequest,
        MaxFlowResult,
        compute_max_flow,
        "graph",
        "flow",
        "max-flow",
        "exact",
        examples=(
            example(
                "simple_max_flow",
                "Compute the maximum flow in a simple graph.",
                {
                    "graph": {
                        "vertex_count": 3,
                        "edges": [
                            {
                                "source": 0,
                                "target": 1,
                                "capacity": {"num": "3", "den": "1"},
                            },
                            {
                                "source": 1,
                                "target": 2,
                                "capacity": {"num": "2", "den": "1"},
                            },
                        ],
                    },
                    "source": 0,
                    "sink": 2,
                },
            ),
            example(
                "four_vertex_max_flow",
                "Compute a maximum flow; edge endpoints, source, and sink must be in 0..vertex_count-1 and source must differ from sink.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "edges": [
                            {
                                "source": 0,
                                "target": 1,
                                "capacity": {"num": "5", "den": "1"},
                            },
                            {
                                "source": 1,
                                "target": 2,
                                "capacity": {"num": "3", "den": "1"},
                            },
                            {
                                "source": 2,
                                "target": 3,
                                "capacity": {"num": "4", "den": "1"},
                            },
                        ],
                    },
                    "source": 0,
                    "sink": 3,
                },
            ),
        ),
    ),
    graph_flow_operation(
        "graph.cut.minimum_st.compute",
        "Compute the minimum s-t cut in a capacitated graph",
        "Compute the minimum s-t cut value and partition in a directed capacitated graph using NetworkX.",
        MinCutRequest,
        MinCutResult,
        compute_min_cut,
        "graph",
        "cut",
        "min-cut",
        "exact",
        examples=(
            example(
                "simple_min_cut",
                "Compute the minimum cut in a simple graph.",
                {
                    "graph": {
                        "vertex_count": 3,
                        "edges": [
                            {
                                "source": 0,
                                "target": 1,
                                "capacity": {"num": "3", "den": "1"},
                            },
                            {
                                "source": 1,
                                "target": 2,
                                "capacity": {"num": "2", "den": "1"},
                            },
                        ],
                    },
                    "source": 0,
                    "sink": 2,
                },
            ),
            example(
                "four_vertex_min_cut",
                "Compute a minimum s-t cut; edge endpoints, source, and sink must be in 0..vertex_count-1 and source must differ from sink.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "edges": [
                            {
                                "source": 0,
                                "target": 1,
                                "capacity": {"num": "5", "den": "1"},
                            },
                            {
                                "source": 1,
                                "target": 2,
                                "capacity": {"num": "3", "den": "1"},
                            },
                            {
                                "source": 2,
                                "target": 3,
                                "capacity": {"num": "4", "den": "1"},
                            },
                        ],
                    },
                    "source": 0,
                    "sink": 3,
                },
            ),
        ),
    ),
    graph_flow_operation(
        "graph.menger.edge_disjoint.compute",
        "Compute the maximum number of edge-disjoint paths (Menger theorem)",
        "Compute the maximum number of edge-disjoint directed paths between source and sink in a simple directed graph using NetworkX, along with the explicit paths.",
        EdgeDisjointPathsRequest,
        EdgeDisjointPathsResult,
        compute_edge_disjoint_paths,
        "graph",
        "menger",
        "edge-disjoint",
        "exact",
        examples=(
            example(
                "two_edge_disjoint_paths",
                "Compute the maximum number of edge-disjoint paths in a diamond graph.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "edges": [
                            [0, 1],
                            [0, 2],
                            [1, 3],
                            [2, 3],
                        ],
                    },
                    "source": 0,
                    "sink": 3,
                },
            ),
        ),
    ),
    graph_flow_operation(
        "network.min_cost_flow.compute",
        "Compute minimum-cost flow with demands",
        "Compute the minimum-cost flow satisfying vertex demands in a "
        "directed graph with capacities and per-unit costs, using "
        "NetworkX's network simplex algorithm.",
        MinCostFlowRequest,
        MinCostFlowResult,
        compute_min_cost_flow,
        "network",
        "min-cost-flow",
        "exact",
        examples=(
            example(
                "simple_min_cost_flow",
                "Send 2 units from node 0 to node 2 via node 1.",
                {
                    "graph": {
                        "vertex_count": 3,
                        "edges": [
                            {
                                "source": 0,
                                "target": 1,
                                "capacity": {"num": "5", "den": "1"},
                                "cost": {"num": "1", "den": "1"},
                            },
                            {
                                "source": 1,
                                "target": 2,
                                "capacity": {"num": "5", "den": "1"},
                                "cost": {"num": "2", "den": "1"},
                            },
                            {
                                "source": 0,
                                "target": 2,
                                "capacity": {"num": "5", "den": "1"},
                                "cost": {"num": "4", "den": "1"},
                            },
                        ],
                    },
                    "demands": [-2, 0, 2],
                },
            ),
        ),
    ),
)

TOOLS = GRAPH_FLOW_OPERATIONS

__all__ = ["TOOLS"]
