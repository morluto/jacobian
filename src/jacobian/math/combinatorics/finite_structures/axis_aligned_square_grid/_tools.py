"""Axis-aligned square grid hypergraph operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def asg_operation[
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
    asg_operation(
        "hypergraph.axis_aligned_square_grid.construct",
        "Construct the axis-aligned-square hypergraph of [N]^2",
        (
            "Construct the 4-uniform hypergraph whose vertices are the N^2 grid "
            "points and whose hyperedges are the axis-aligned squares "
            "{(x,y), (x+d,y), (x,y+d), (x+d,y+d)} for every d >= 1."
        ),
        AxisAlignedSquareGridRequest,
        AxisAlignedSquareGridResult,
        compute_axis_aligned_square_grid,
        "combinatorics",
        "hypergraph",
        "exact",
        examples=(
            example(
                "n3",
                "The 3x3 grid has 9 vertices and 5 axis-aligned squares; "
                "side_length must be an integer from 1 through 16.",
                {"side_length": 3},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
