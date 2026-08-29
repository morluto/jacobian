"""Edge-colouring Ramsey arrowing operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.edge_coloring_arrowing._models import (
    EdgeColoringArrowingRequest,
    EdgeColoringArrowingResult,
)
from jacobian.math.graphs.edge_coloring_arrowing.operations import (
    decide_edge_coloring_arrowing,
)


def compute_edge_coloring_arrowing(
    request: EdgeColoringArrowingRequest,
) -> EdgeColoringArrowingResult:
    return decide_edge_coloring_arrowing(request.host_graph, request.targets)


def eca_operation[
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
    eca_operation(
        "graph.edge_coloring_arrowing.decide",
        "Decide bounded edge-colouring Ramsey arrowing",
        (
            "Given a bounded host graph and an ordered tuple of target graphs "
            "(one per edge colour), decide whether every edge-colouring of the "
            "host contains a monochromatic copy of the corresponding target in "
            "some colour. Returns ARROWS or DOES_NOT_ARROW with one avoiding "
            "colouring."
        ),
        EdgeColoringArrowingRequest,
        EdgeColoringArrowingResult,
        compute_edge_coloring_arrowing,
        "graph",
        "ramsey",
        "exact",
        examples=(
            example(
                "k6_arrows_k3_k3",
                "K6 arrows (K3,K3) under red/blue edge colourings.",
                {
                    "host_graph": {
                        "vertices": ["0", "1", "2", "3", "4", "5"],
                        "edges": [
                            ["0", "1"],
                            ["0", "2"],
                            ["0", "3"],
                            ["0", "4"],
                            ["0", "5"],
                            ["1", "2"],
                            ["1", "3"],
                            ["1", "4"],
                            ["1", "5"],
                            ["2", "3"],
                            ["2", "4"],
                            ["2", "5"],
                            ["3", "4"],
                            ["3", "5"],
                            ["4", "5"],
                        ],
                    },
                    "targets": [
                        {
                            "vertices": ["0", "1", "2"],
                            "edges": [["0", "1"], ["1", "2"], ["0", "2"]],
                        },
                        {
                            "vertices": ["0", "1", "2"],
                            "edges": [["0", "1"], ["1", "2"], ["0", "2"]],
                        },
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
