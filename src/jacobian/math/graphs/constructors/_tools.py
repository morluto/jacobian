"""Graph constructor operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.constructors._models import (
    HypercubeGraphRequest,
    HypercubeGraphResult,
    KellerGraphRequest,
    KellerGraphResult,
    TriangleProfileRequest,
    TriangleProfileResult,
)
from jacobian.math.graphs.constructors._operations import (
    _run_hypercube_graph,
    _run_keller_graph,
    _run_triangle_profile,
)


def gt_operation[RequestT: StrictModel, ResultT: StrictModel](
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
    gt_operation(
        "graph.hypercube.construct",
        "Construct hypercube graph",
        (
            "Construct the d-dimensional hypercube graph Q_d whose vertices "
            "are binary strings of length d and whose edges connect vertices "
            "differing in exactly one bit."
        ),
        HypercubeGraphRequest,
        HypercubeGraphResult,
        _run_hypercube_graph,
        "graph",
        "constructor",
        "hypercube",
        examples=(
            example(
                "hypercube_d3",
                "Construct the 3-dimensional hypercube Q_3 with 8 vertices; the dimension must be at most 8.",
                {"dimension": 3},
            ),
        ),
    ),
    gt_operation(
        "graph.keller.construct",
        "Construct Keller graph",
        (
            "Construct the Keller graph K_d whose vertices are words in "
            "{0,1,2,3}^d, with two distinct words adjacent iff they differ "
            "by 2 (mod 4) in at least one coordinate and differ in at least "
            "two coordinates overall."
        ),
        KellerGraphRequest,
        KellerGraphResult,
        _run_keller_graph,
        "graph",
        "constructor",
        "keller",
        examples=(
            example(
                "keller_d2",
                "Construct the Keller graph K_2 with 16 vertices; the dimension must be at most 4.",
                {"dimension": 2},
            ),
        ),
    ),
    gt_operation(
        "graph.triangle_profile.compute",
        "Compute triangle profile",
        (
            "Return the complete list of triangles in a finite simple "
            "undirected graph, each as an ordered triple of vertex labels."
        ),
        TriangleProfileRequest,
        TriangleProfileResult,
        _run_triangle_profile,
        "graph",
        "triangle",
        "profile",
        examples=(
            example(
                "triangle_profile_k4",
                "Compute all four triangles of K_4; the source must be a canonical finite simple undirected graph.",
                {
                    "graph": {
                        "vertices": ["a", "b", "c", "d"],
                        "edges": [
                            ["a", "b"],
                            ["a", "c"],
                            ["a", "d"],
                            ["b", "c"],
                            ["b", "d"],
                            ["c", "d"],
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
