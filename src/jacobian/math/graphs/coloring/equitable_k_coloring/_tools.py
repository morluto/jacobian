"""Typed declarations for the equitable k-colouring decision."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.coloring.equitable_k_coloring._models import (
    EquitableColoringRequest,
    EquitableColoringResult,
)
from jacobian.math.graphs.coloring.equitable_k_coloring.operations import (
    decide_equitable_k_coloring,
)


def _decide(request: EquitableColoringRequest) -> EquitableColoringResult:
    return decide_equitable_k_coloring(request.graph, request.k)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.coloring.equitable_k_colorability.decide",
        title="Decide bounded equitable k-colourability",
        description=(
            "Given a bounded indexed simple graph G and positive integer k, "
            "decide whether G has a proper k-colouring in which every used "
            "colour class has size either floor(|V|/k) or ceil(|V|/k)."
        ),
        request_type=EquitableColoringRequest,
        result_type=EquitableColoringResult,
        run=_decide,
        tags=("graph", "coloring", "equitable", "exact"),
        examples=(
            OperationExample(
                name="k4_complete",
                description="Equitable 4-colouring of K4.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c", "d"],
                        "edges": [
                            ["a", "b"],
                            ["a", "c"],
                            ["a", "d"],
                            ["b", "c"],
                            ["b", "d"],
                            ["c", "d"],
                        ],
                    },
                    "k": 4,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
