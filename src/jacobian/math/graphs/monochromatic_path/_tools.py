"""Monochromatic path hypergraph operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.monochromatic_path._models import (
    MonochromaticPathRequest,
    MonochromaticPathResult,
)
from jacobian.math.graphs.monochromatic_path.operations import (
    construct_monochromatic_path_hypergraphs,
)


def compute_monochromatic_path_op(
    request: MonochromaticPathRequest,
) -> MonochromaticPathResult:
    return construct_monochromatic_path_hypergraphs(request.graph)


def mph_action[RequestT: StrictModel, ResultT: StrictModel](
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


TOOLS: MathTools = (
    mph_action(
        "graph.edge_colored.monochromatic_path_hypergraphs.construct",
        "Construct monochromatic path hypergraphs from a coloured graph",
        (
            "For each colour in an edge-coloured graph, return one canonical "
            "FiniteHypergraph whose hyperedges are the nonempty source-vertex "
            "sets that admit a simple path using only edges of that colour."
        ),
        MonochromaticPathRequest,
        MonochromaticPathResult,
        compute_monochromatic_path_op,
        "graph",
        "ramsey",
        "exact",
        examples=(
            example(
                "all_red_k3",
                "All-red K3: red path on every nonempty vertex subset.",
                {
                    "graph": {
                        "graph": {
                            "vertices": ["0", "1", "2"],
                            "edges": [["0", "1"], ["0", "2"], ["1", "2"]],
                        },
                        "edge_colors": ["red", "red", "red"],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
