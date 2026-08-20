"""Finite simplicial topology domain."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.topology._models import (
    FVectorRequest,
    FVectorResult,
    LinkRequest,
    LinkResult,
)
from jacobian.math.topology._operations import (
    TOPOLOGY_OPERATIONS,
    compute_f_vector,
    compute_link,
)

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

_link_tool = MathTool(
    operation_id="topology.simplicial_complex.link.compute",
    version="1",
    title="Compute the link of a simplex",
    description=(
        "Compute the link of a simplex in a finite simplicial complex and return "
        "the maximal facets of the resulting link complex."
    ),
    request_type=LinkRequest,
    result_type=LinkResult,
    run=compute_link,
    tags=("topology", "simplicial", "exact"),
    examples=(
        example(
            "link_of_vertex_in_triangle",
            "Compute the link of one vertex in a triangle.",
            {
                "complex": {
                    "vertices": ["v0", "v1", "v2"],
                    "facets": [["v0", "v1", "v2"]],
                },
                "simplex": ["v0"],
            },
        ),
    ),
)

TOOLS: MathTools = (*TOPOLOGY_OPERATIONS, _f_vector_tool, _link_tool)
