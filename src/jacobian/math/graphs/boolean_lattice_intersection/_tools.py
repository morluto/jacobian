"""Typed declarations for the Boolean-lattice intersection graph operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.boolean_lattice_intersection._models import (
    BooleanLatticeIntersectionRequest,
    BooleanLatticeIntersectionResult,
)
from jacobian.math.graphs.boolean_lattice_intersection.operations import (
    construct_boolean_lattice_intersection_graph,
)


def _construct(request: BooleanLatticeIntersectionRequest) -> BooleanLatticeIntersectionResult:
    return construct_boolean_lattice_intersection_graph(
        request.n, request.intersection_cardinality, request.relation
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.boolean_lattice_intersection.construct",
        title="Construct a Boolean-lattice intersection graph",
        description=(
            "Given a nonnegative n and an intersection cardinality r, construct "
            "a simple graph with one vertex for each subset of [n] and an edge "
            "exactly when the two source subsets satisfy the declared intersection "
            "relation."
        ),
        request_type=BooleanLatticeIntersectionRequest,
        result_type=BooleanLatticeIntersectionResult,
        run=_construct,
        tags=("graph", "combinatorics", "boolean", "lattice", "exact"),
        examples=(
            example(
                "n2_r1",
                "Boolean lattice intersection graph for n=2, r=1.",
                {"n": 2, "intersection_cardinality": 1, "relation": "INTERSECTION_EQ_THRESHOLD"},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
