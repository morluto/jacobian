"""Finite simplicial topology domain."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.topology._models import FVectorRequest, FVectorResult
from jacobian.math.topology._operations import TOPOLOGY_OPERATIONS, compute_f_vector

__all__ = ["TOOLS"]

_f_vector_tool = MathTool(
    operation_id="topology.simplicial_complex.f_vector.compute",
    version="1",
    title="Compute the f-vector and h-vector of a simplicial complex",
    description=(
        "Compute the f-vector (face counts by dimension) and h-vector "
        "of a finite simplicial complex, with Euler characteristic."
    ),
    request_type=FVectorRequest,
    result_type=FVectorResult,
    run=compute_f_vector,
    tags=("topology", "simplicial", "exact"),
    examples=(
        example(
            "triangle_f_vector",
            "Compute f-vector of a triangle (3 vertices, 3 edges, 1 face); "
            "facets must be a list of simplices.",
            {
                "complex": {
                    "vertices": ["v0", "v1", "v2"],
                    "facets": [["v0", "v1", "v2"]],
                }
            },
        ),
    ),
)

TOOLS: MathTools = (*TOPOLOGY_OPERATIONS, _f_vector_tool)
