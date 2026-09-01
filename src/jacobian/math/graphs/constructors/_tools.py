"""Graph constructor operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.constructors._models import (
    HypercubeGraphRequest,
    HypercubeGraphResult,
    KellerGraphRequest,
    KellerGraphResult,
    TriangleProfileRequest,
    TriangleProfileResult,
)
from jacobian.math.graphs.constructors.operations import (
    compute_triangle_profile,
    construct_hypercube_graph,
    construct_keller_graph,
)


def _run_hypercube_graph(request: HypercubeGraphRequest) -> HypercubeGraphResult:
    return construct_hypercube_graph(request.dimension)


def _run_keller_graph(request: KellerGraphRequest) -> KellerGraphResult:
    return construct_keller_graph(request.dimension)


def _run_triangle_profile(request: TriangleProfileRequest) -> TriangleProfileResult:
    return compute_triangle_profile(request.graph)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.hypercube.construct",
        title="Construct hypercube graph",
        description=(
            "Construct the d-dimensional hypercube graph Q_d whose vertices "
            "are binary strings of length d and whose edges connect vertices "
            "differing in exactly one bit."
        ),
        request_type=HypercubeGraphRequest,
        result_type=HypercubeGraphResult,
        run=_run_hypercube_graph,
        tags=("graph", "constructor", "hypercube"),
        examples=(
            OperationExample(
                name="hypercube_d3",
                description="Construct the 3-dimensional hypercube Q_3 with 8 vertices; the dimension must be at most 8.",
                input={"dimension": 3},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.keller.construct",
        title="Construct Keller graph",
        description=(
            "Construct the Keller graph K_d whose vertices are words in "
            "{0,1,2,3}^d, with two distinct words adjacent iff they differ "
            "by 2 (mod 4) in at least one coordinate and differ in at least "
            "two coordinates overall."
        ),
        request_type=KellerGraphRequest,
        result_type=KellerGraphResult,
        run=_run_keller_graph,
        tags=("graph", "constructor", "keller"),
        examples=(
            OperationExample(
                name="keller_d2",
                description="Construct the Keller graph K_2 with 16 vertices; the dimension must be at most 4.",
                input={"dimension": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.triangle_profile.compute",
        title="Compute triangle profile",
        description=(
            "Return the complete list of triangles in a finite simple "
            "undirected graph, each as an ordered triple of vertex labels."
        ),
        request_type=TriangleProfileRequest,
        result_type=TriangleProfileResult,
        run=_run_triangle_profile,
        tags=("graph", "triangle", "profile"),
        examples=(
            OperationExample(
                name="triangle_profile_k4",
                description="Compute all four triangles of K_4; the source must be a canonical finite simple undirected graph.",
                input={
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
