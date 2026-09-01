"""Monochromatic path hypergraph operation declarations."""

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


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.edge_colored.monochromatic_path_hypergraphs.construct",
        title="Construct monochromatic path hypergraphs from a coloured graph",
        description=(
            "For each colour in an edge-coloured graph, return one canonical "
            "FiniteHypergraph whose hyperedges are the nonempty source-vertex "
            "sets that admit a simple path using only edges of that colour."
        ),
        request_type=MonochromaticPathRequest,
        result_type=MonochromaticPathResult,
        run=compute_monochromatic_path_op,
        tags=("graph", "ramsey", "exact"),
        examples=(
            OperationExample(
                name="all_red_k3",
                description="All-red K3: red path on every nonempty vertex subset.",
                input={
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
