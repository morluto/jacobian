"""Uniform-subset intersection graph operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def usi_operation[
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
    usi_operation(
        "graph.uniform_subset_intersection.construct",
        "Construct a uniform-subset intersection graph",
        (
            "Construct a labelled simple graph whose vertices are all k-subsets "
            "of the zero-based ground set {0,...,n-1}, with an edge between two "
            "subsets when their intersection size "
            "satisfies the declared less-than-threshold or equality relation. "
            "Includes Kneser graphs and Johnson-scheme threshold graphs "
            "as special cases."
        ),
        UniformSubsetIntersectionRequest,
        UniformSubsetIntersectionResult,
        compute_uniform_subset_intersection_graph,
        "graph",
        "combinatorics",
        "exact",
        examples=(
            example(
                "kneser_kg42",
                "Kneser graph KG(4,2): 2-subsets of {0,1,2,3} with intersection < 1.",
                {
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
