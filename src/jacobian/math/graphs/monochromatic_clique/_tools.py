"""Monochromatic clique hypergraph operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.monochromatic_clique._models import (
    MonochromaticCliqueHypergraphRequest,
    MonochromaticCliqueHypergraphResult,
)
from jacobian.math.graphs.monochromatic_clique.operations import (
    construct_monochromatic_clique_hypergraph,
)


def compute_monochromatic_clique_hypergraph(
    request: MonochromaticCliqueHypergraphRequest,
) -> MonochromaticCliqueHypergraphResult:
    return construct_monochromatic_clique_hypergraph(
        request.colored_graph, request.clique_order
    )


def mc_operation[
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
    mc_operation(
        "graph.edge_colored.monochromatic_clique_hypergraph.construct",
        "Construct the monochromatic-clique hypergraph of an edge-coloured graph",
        (
            "Given a bounded complete edge-coloured simple graph and an integer "
            "t >= 2, return the canonical t-uniform FiniteHypergraph whose "
            "hyperedges are exactly the t-element vertex sets inducing a "
            "monochromatic clique in the source."
        ),
        MonochromaticCliqueHypergraphRequest,
        MonochromaticCliqueHypergraphResult,
        compute_monochromatic_clique_hypergraph,
        "graph",
        "ramsey",
        "exact",
        examples=(
            example(
                "all_red_k4_t3",
                "All-red K4 with target clique order 3.",
                {
                    "colored_graph": {
                        "graph": {
                            "vertices": ["0", "1", "2", "3"],
                            "edges": [
                                ["0", "1"],
                                ["0", "2"],
                                ["0", "3"],
                                ["1", "2"],
                                ["1", "3"],
                                ["2", "3"],
                            ],
                        },
                        "edge_colors": ["red", "red", "red", "red", "red", "red"],
                    },
                    "clique_order": 3,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
