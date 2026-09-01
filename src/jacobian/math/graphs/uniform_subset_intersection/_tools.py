"""Uniform-subset intersection graph operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.uniform_subset_intersection._models import (
    UniformSubsetIntersectionRequest,
    UniformSubsetIntersectionResult,
)
from jacobian.math.graphs.uniform_subset_intersection.operations import (
    construct_uniform_subset_intersection_graph,
)


def compute_uniform_subset_intersection_graph(
    request: UniformSubsetIntersectionRequest,
) -> UniformSubsetIntersectionResult:
    return construct_uniform_subset_intersection_graph(
        request.ground_set_size,
        request.subset_cardinality,
        request.threshold,
        request.relation,
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.uniform_subset_intersection.construct",
        title="Construct a uniform-subset intersection graph",
        description=(
            "Construct a labelled simple graph whose vertices are all k-subsets "
            "of the zero-based ground set {0,...,n-1}, with an edge between two "
            "subsets when their intersection size "
            "satisfies the declared less-than-threshold or equality relation. "
            "Includes Kneser graphs and Johnson-scheme threshold graphs "
            "as special cases."
        ),
        request_type=UniformSubsetIntersectionRequest,
        result_type=UniformSubsetIntersectionResult,
        run=compute_uniform_subset_intersection_graph,
        tags=("graph", "combinatorics", "exact"),
        examples=(
            OperationExample(
                name="kneser_kg42",
                description="Kneser graph KG(4,2): 2-subsets of {0,1,2,3} with intersection < 1.",
                input={
                    "ground_set_size": 4,
                    "subset_cardinality": 2,
                    "threshold": 1,
                    "relation": "INTERSECTION_LT_THRESHOLD",
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
