"""Typed declarations for the uniform-subset intersection graph operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.uniform_subset_intersection._models import (
    UniformSubsetIntersectionRequest,
    UniformSubsetIntersectionResult,
)
from jacobian.math.graphs.uniform_subset_intersection.operations import (
    construct_uniform_subset_intersection_graph,
)


def _construct(
    request: UniformSubsetIntersectionRequest,
) -> UniformSubsetIntersectionResult:
    return construct_uniform_subset_intersection_graph(
        request.n, request.k, request.threshold, request.relation
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.uniform_subset_intersection.construct",
        title="Construct a uniform-subset intersection graph",
        description=(
            "Given ground-set size n, subset cardinality k, and a threshold t, "
            "construct a simple graph with one vertex for every k-subset of [n] "
            "and an edge exactly when the intersection-size relation holds."
        ),
        request_type=UniformSubsetIntersectionRequest,
        result_type=UniformSubsetIntersectionResult,
        run=_construct,
        tags=("graph", "combinatorics", "intersection", "exact"),
        examples=(
            example(
                "kneser_5_2",
                "Kneser graph KG(5,2) with t=0 (disjoint).",
                {
                    "n": 5,
                    "k": 2,
                    "threshold": 0,
                    "relation": "INTERSECTION_EQ_THRESHOLD",
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
