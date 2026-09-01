"""Boolean-lattice intersection graph operation declarations."""

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


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.boolean_lattice_intersection.construct",
        title="Construct a Boolean-lattice intersection graph",
        description=(
            "Construct a labelled simple graph whose vertices are all subsets "
            "of [n], with an edge between two subsets when their intersection "
            "size satisfies the declared relation (equal, less-than, or "
            "greater-than) with a given threshold."
        ),
        request_type=BooleanLatticeIntersectionRequest,
        result_type=BooleanLatticeIntersectionResult,
        run=compute_boolean_lattice_intersection,
        tags=("graph", "combinatorics", "exact"),
        examples=(
            OperationExample(
                name="eq_r0_n2",
                description="Boolean lattice 2^[2] with intersection == 0.",
                input={
                    "ground_set_size": 2,
                    "threshold": 0,
                    "relation": "INTERSECTION_EQ",
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
