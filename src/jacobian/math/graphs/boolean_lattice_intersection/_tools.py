"""Boolean-lattice intersection graph operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.boolean_lattice_intersection._models import (
    BooleanLatticeIntersectionRequest,
    BooleanLatticeIntersectionResult,
)
from jacobian.math.graphs.boolean_lattice_intersection.operations import (
    construct_boolean_lattice_intersection_graph,
)


def compute_boolean_lattice_intersection(
    request: BooleanLatticeIntersectionRequest,
) -> BooleanLatticeIntersectionResult:
    return construct_boolean_lattice_intersection_graph(
        request.ground_set_size, request.threshold, request.relation
    )


def bli_operation[
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
    bli_operation(
        "graph.boolean_lattice_intersection.construct",
        "Construct a Boolean-lattice intersection graph",
        (
            "Construct a labelled simple graph whose vertices are all subsets "
            "of [n], with an edge between two subsets when their intersection "
            "size satisfies the declared relation (equal, less-than, or "
            "greater-than) with a given threshold."
        ),
        BooleanLatticeIntersectionRequest,
        BooleanLatticeIntersectionResult,
        compute_boolean_lattice_intersection,
        "graph",
        "combinatorics",
        "exact",
        examples=(
            example(
                "eq_r0_n2",
                "Boolean lattice 2^[2] with intersection == 0.",
                {
                    "ground_set_size": 2,
                    "threshold": 0,
                    "relation": "INTERSECTION_EQ",
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
