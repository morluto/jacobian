"""Axis-aligned square grid hypergraph operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.finite_structures.axis_aligned_square_grid._models import (
    AxisAlignedSquareGridRequest,
    AxisAlignedSquareGridResult,
)
from jacobian.math.combinatorics.finite_structures.axis_aligned_square_grid.operations import (
    construct_axis_aligned_square_grid,
)


def compute_axis_aligned_square_grid(
    request: AxisAlignedSquareGridRequest,
) -> AxisAlignedSquareGridResult:
    return construct_axis_aligned_square_grid(request.side_length)


TOOLS: MathTools = (
    MathTool(
        operation_id="hypergraph.axis_aligned_square_grid.construct",
        title="Construct the axis-aligned-square hypergraph of [N]^2",
        description=(
            "Construct the 4-uniform hypergraph whose vertices are the N^2 grid "
            "points and whose hyperedges are the axis-aligned squares "
            "{(x,y), (x+d,y), (x,y+d), (x+d,y+d)} for every d >= 1."
        ),
        request_type=AxisAlignedSquareGridRequest,
        result_type=AxisAlignedSquareGridResult,
        run=compute_axis_aligned_square_grid,
        tags=("combinatorics", "hypergraph", "exact"),
        examples=(
            OperationExample(
                name="n3",
                description="The 3x3 grid has 9 vertices and 5 axis-aligned squares; "
                "side_length must be an integer from 1 through 16.",
                input={"side_length": 3},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
