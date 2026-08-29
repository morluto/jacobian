"""Chip-firing operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.chip_firing import operations as native
from jacobian.math.graphs.chip_firing._models import (
    AbelJacobiRequest,
    AbelJacobiResult,
    CanonicalDivisorRequest,
    CanonicalDivisorResult,
    CriticalGroupRequest,
    CriticalGroupResult,
    DegreeRequest,
    DegreeResult,
    FireVectorRequest,
    FireVectorResult,
    FiringRequest,
    FiringResult,
    LaplacianRequest,
    LaplacianResult,
    ParallelStepRequest,
    ParallelStepResult,
    QReducedRequest,
    QReducedResult,
    ReducedLaplacianRequest,
    ReducedLaplacianResult,
    StabilizeRequest,
    StabilizeResult,
)


def compute_laplacian(request: LaplacianRequest) -> LaplacianResult:
    return native.laplacian(request.graph)


def compute_reduced_laplacian(
    request: ReducedLaplacianRequest,
) -> ReducedLaplacianResult:
    return native.reduced_laplacian(request.graph, request.sink)


def compute_firing(request: FiringRequest) -> FiringResult:
    return native.firing(request.graph, request.divisor, request.firing_vertex)


def compute_fire_vector(request: FireVectorRequest) -> FireVectorResult:
    return native.fire_vector(request.graph, request.divisor, request.firing_vector)


def compute_stabilize(request: StabilizeRequest) -> StabilizeResult:
    configuration = request.configuration
    return native.stabilize(
        configuration.graph, configuration.sink, configuration.configuration
    )


def compute_parallel_step(request: ParallelStepRequest) -> ParallelStepResult:
    configuration = request.configuration
    return native.parallel_step(
        configuration.graph, configuration.sink, configuration.configuration
    )


def compute_q_reduced(request: QReducedRequest) -> QReducedResult:
    return native.q_reduced(request.graph, request.divisor, request.sink)


def compute_canonical_divisor(
    request: CanonicalDivisorRequest,
) -> CanonicalDivisorResult:
    return native.canonical_divisor(request.graph)


def compute_critical_group(request: CriticalGroupRequest) -> CriticalGroupResult:
    return native.critical_group(request.graph, request.sink)


def compute_degree(request: DegreeRequest) -> DegreeResult:
    return native.degree(request.divisor)


def compute_abel_jacobi(request: AbelJacobiRequest) -> AbelJacobiResult:
    return native.abel_jacobi(request.graph, request.divisor, request.sink)


def _op[RequestT: StrictModel, ResultT: StrictModel](
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


_GRAPH = {"vertices": ["a", "b", "c"], "edges": [["a", "b"], ["b", "c"]]}
_SINK_CONFIG = {
    "graph": _GRAPH,
    "sink": "a",
    "configuration": [0, 3, 0],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "graph.chip_firing.laplacian.compute",
        "Compute the graph Laplacian",
        "Compute the exact graph Laplacian matrix L = D - A where D is "
        "the degree matrix and A is the adjacency matrix, with vertex "
        "labels and degree vector.",
        LaplacianRequest,
        LaplacianResult,
        compute_laplacian,
        "graph-theory",
        "chip-firing",
        "exact",
        examples=(
            example(
                "path_graph_3",
                "Compute the Laplacian of a path graph on 3 vertices; "
                "the graph must be a finite undirected simple graph.",
                {"graph": _GRAPH},
            ),
        ),
    ),
    _op(
        "graph.chip_firing.reduced_laplacian.compute",
        "Compute the reduced Laplacian",
        "Delete the sink row and column from the full Laplacian and "
        "return the labelled reduced Laplacian with nonsink vertex "
        "correspondence.",
        ReducedLaplacianRequest,
        ReducedLaplacianResult,
        compute_reduced_laplacian,
        "graph-theory",
        "chip-firing",
        "exact",
        examples=(
            example(
                "path_graph_3_sink_a",
                "Compute the reduced Laplacian of a path graph with sink at vertex a.",
                {"graph": _GRAPH, "sink": "a"},
            ),
        ),
    ),
    _op(
        "graph.chip_firing.fire_vertex.compute",
        "Fire a vertex in a chip configuration",
        "Fire a vertex v in a chip configuration: v loses degree(v) "
        "chips and each neighbor gains one chip per edge. Returns "
        "the transformed divisor.",
        FiringRequest,
        FiringResult,
        compute_firing,
        "graph-theory",
        "chip-firing",
        "exact",
        examples=(
            example(
                "fire_vertex_b",
                "Fire vertex b in a path graph; "
                "the divisor length must match the vertex count.",
                {"graph": _GRAPH, "divisor": [3, 0, 1], "firing_vertex": "b"},
            ),
        ),
    ),
    _op(
        "graph.chip_firing.fire_vector.compute",
        "Fire a vector in a chip configuration",
        "Apply an integer firing vector f to a divisor: D' = D - L f. "
        "Degree is preserved by construction.",
        FireVectorRequest,
        FireVectorResult,
        compute_fire_vector,
        "graph-theory",
        "chip-firing",
        "exact",
        examples=(
            example(
                "fire_e_a",
                "Fire the unit vector e_a on a path graph; degree is preserved.",
                {
                    "graph": _GRAPH,
                    "divisor": [3, 0, 1],
                    "firing_vector": [1, 0, 0],
                },
            ),
        ),
    ),
    _op(
        "graph.chip_firing.stabilize.compute",
        "Stabilize a sink configuration",
        "Stabilize a bounded sink configuration and return the unique "
        "stable configuration, exact odometer (toppling-count) vector, "
        "and total firing count.",
        StabilizeRequest,
        StabilizeResult,
        compute_stabilize,
        "graph-theory",
        "chip-firing",
        "exact",
        examples=(
            example(
                "path_graph_3_sink_a",
                "Stabilize a path graph configuration with sink at vertex a.",
                {"configuration": _SINK_CONFIG},
            ),
        ),
    ),
    _op(
        "graph.chip_firing.parallel_step.compute",
        "One parallel firing step",
        "Apply one simultaneous legal firing step to every currently "
        "unstable nonsink vertex and return the next configuration "
        "plus the fired vertex set.",
        ParallelStepRequest,
        ParallelStepResult,
        compute_parallel_step,
        "graph-theory",
        "chip-firing",
        "exact",
        examples=(
            example(
                "path_graph_3_sink_a",
                "Apply one parallel step on a path graph configuration "
                "with sink at vertex a.",
                {"configuration": _SINK_CONFIG},
            ),
        ),
    ),
    _op(
        "graph.chip_firing.q_reduced.compute",
        "q-reduced normal form",
        "Compute the unique q-reduced representative of a graph "
        "divisor under the standard connected-graph convention, plus "
        "the exact firing vector f satisfying D_reduced = D - L f.",
        QReducedRequest,
        QReducedResult,
        compute_q_reduced,
        "graph-theory",
        "chip-firing",
        "exact",
        examples=(
            example(
                "triangle_sink_a",
                "Compute the q-reduced form of a divisor on a triangle "
                "graph with sink at vertex a.",
                {
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                    },
                    "divisor": [5, 0, 0],
                    "sink": "a",
                },
            ),
        ),
    ),
    _op(
        "graph.chip_firing.canonical_divisor.compute",
        "Graph canonical divisor",
        "Compute the graph canonical divisor K(v) = degree(v) - 2 "
        "and its exact degree 2|E| - 2|V|.",
        CanonicalDivisorRequest,
        CanonicalDivisorResult,
        compute_canonical_divisor,
        "graph-theory",
        "chip-firing",
        "exact",
        examples=(
            example(
                "path_graph_3",
                "Compute the canonical divisor of a path graph on 3 vertices.",
                {"graph": _GRAPH},
            ),
        ),
    ),
    _op(
        "graph.chip_firing.critical_group.compute",
        "Critical group (sandpile group)",
        "Compute the critical group of a connected graph via Smith "
        "normal form of the reduced Laplacian. Returns invariant "
        "factors and group order.",
        CriticalGroupRequest,
        CriticalGroupResult,
        compute_critical_group,
        "graph-theory",
        "chip-firing",
        "exact",
        examples=(
            example(
                "triangle_sink_a",
                "Compute the critical group of a triangle graph with sink at vertex a.",
                {
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                    },
                    "sink": "a",
                },
            ),
        ),
    ),
    _op(
        "graph.chip_firing.abel_jacobi.compute",
        "Abel-Jacobi coordinates",
        "Map a degree-zero graph divisor into the critical group via "
        "the Abel-Jacobi map, returning canonical class coordinates in "
        "the cokernel of the reduced Laplacian.",
        AbelJacobiRequest,
        AbelJacobiResult,
        compute_abel_jacobi,
        "graph-theory",
        "chip-firing",
        "exact",
        examples=(
            example(
                "triangle_sink_a",
                "Map a degree-zero divisor on a triangle graph with "
                "sink at vertex a into the critical group.",
                {
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                    },
                    "divisor": [1, -1, 0],
                    "sink": "a",
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
