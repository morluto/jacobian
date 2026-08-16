"""Exact graph isomorphism operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.graph_isomorphism import (
    GraphIsomorphismRequest,
    GraphIsomorphismResult,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.graph_isomorphism.operations import (
    compute_isomorphism_decision,
)
from jacobian.math_tools import MathTool


def graph_isomorphism_operation[
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


GRAPH_ISOMORPHISM_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    graph_isomorphism_operation(
        "graph.isomorphism.decide",
        "Decide graph isomorphism with explicit mapping certificate",
        "Decide whether two simple undirected graphs are isomorphic using "
        "NetworkX. Returns ISOMORPHIC with an explicit vertex mapping "
        "certificate, or NOT_ISOMORPHIC.",
        GraphIsomorphismRequest,
        GraphIsomorphismResult,
        compute_isomorphism_decision,
        "graph",
        "isomorphism",
        "structure",
        "compare",
        "decision",
        "exact",
        examples=(
            example(
                "path_graph_isomorphic",
                "Decide isomorphism of two 3-vertex path graphs.",
                {
                    "graph_a": {
                        "vertex_count": 3,
                        "edges": [[0, 1], [1, 2]],
                    },
                    "graph_b": {
                        "vertex_count": 3,
                        "edges": [[0, 1], [1, 2]],
                    },
                },
            ),
        ),
    ),
)
