"""Exact graph flow operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.graph_flow import (
    MaxFlowRequest,
    MaxFlowResult,
    MinCutRequest,
    MinCutResult,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.graph_flow.operations import compute_max_flow, compute_min_cut
from jacobian.math_tools import MathTool


def graph_flow_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
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
        "Compute the maximum flow value between source and sink in a directed capacitated graph using NetworkX.",
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
)
