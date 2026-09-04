"""Chip-firing operation declarations."""

from typing import Any

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


_GRAPH = {"vertices": ["a", "b", "c"], "edges": [["a", "b"], ["b", "c"]]}
_SINK_CONFIG = {
    "graph": _GRAPH,
    "sink": "a",
    "configuration": [0, 3, 0],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.chip_firing.laplacian.compute",
        title="Compute the graph Laplacian",
        description="Compute the exact graph Laplacian matrix L = D - A where D is "
        "the degree matrix and A is the adjacency matrix, with vertex "
        "labels and degree vector.",
        request_type=LaplacianRequest,
        result_type=LaplacianResult,
        run=compute_laplacian,
        tags=("graph-theory", "chip-firing", "exact"),
        examples=(
            OperationExample(
                name="path_graph_3",
                description="Compute the Laplacian of a path graph on 3 vertices; "
                "the graph must be a finite undirected simple graph.",
                input={"graph": _GRAPH},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.chip_firing.reduced_laplacian.compute",
        title="Compute the reduced Laplacian",
        description="Delete the sink row and column from the full Laplacian and "
        "return the labelled reduced Laplacian with nonsink vertex "
        "correspondence.",
        request_type=ReducedLaplacianRequest,
        result_type=ReducedLaplacianResult,
        run=compute_reduced_laplacian,
        tags=("graph-theory", "chip-firing", "exact"),
        examples=(
            OperationExample(
                name="path_graph_3_sink_a",
                description="Compute the reduced Laplacian of a path graph with sink at vertex a.",
                input={"graph": _GRAPH, "sink": "a"},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.chip_firing.fire_vertex.compute",
        title="Fire a vertex in a chip configuration",
        description="Fire a vertex v in a chip configuration: v loses degree(v) "
        "chips and each neighbor gains one chip per edge. Returns "
        "the transformed divisor.",
        request_type=FiringRequest,
        result_type=FiringResult,
        run=compute_firing,
        tags=("graph-theory", "chip-firing", "exact"),
        examples=(
            OperationExample(
                name="fire_vertex_b",
                description="Fire vertex b in a path graph; "
                "the divisor length must match the vertex count.",
                input={"graph": _GRAPH, "divisor": [3, 0, 1], "firing_vertex": "b"},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.chip_firing.fire_vector.compute",
        title="Fire a vector in a chip configuration",
        description="Apply an integer firing vector f to a divisor: D' = D - L f. "
        "Degree is preserved by construction.",
        request_type=FireVectorRequest,
        result_type=FireVectorResult,
        run=compute_fire_vector,
        tags=("graph-theory", "chip-firing", "exact"),
        examples=(
            OperationExample(
                name="fire_e_a",
                description="Fire the unit vector e_a on a path graph; degree is preserved.",
                input={
                    "graph": _GRAPH,
                    "divisor": [3, 0, 1],
                    "firing_vector": [1, 0, 0],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.chip_firing.stabilize.compute",
        title="Stabilize a sink configuration",
        description="Stabilize a bounded sink configuration on a connected graph and return the unique "
        "stable configuration, exact odometer (toppling-count) vector, "
        "and total firing count.",
        request_type=StabilizeRequest,
        result_type=StabilizeResult,
        run=compute_stabilize,
        tags=("graph-theory", "chip-firing", "exact"),
        examples=(
            OperationExample(
                name="path_graph_3_sink_a",
                description="Stabilize a path graph configuration with sink at vertex a.",
                input={"configuration": _SINK_CONFIG},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.chip_firing.parallel_step.compute",
        title="One parallel firing step",
        description="Apply one simultaneous legal firing step to every currently "
        "unstable nonsink vertex and return the next configuration "
        "plus the fired vertex set.",
        request_type=ParallelStepRequest,
        result_type=ParallelStepResult,
        run=compute_parallel_step,
        tags=("graph-theory", "chip-firing", "exact"),
        examples=(
            OperationExample(
                name="path_graph_3_sink_a",
                description="Apply one parallel step on a path graph configuration "
                "with sink at vertex a.",
                input={"configuration": _SINK_CONFIG},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.chip_firing.q_reduced.compute",
        title="q-reduced normal form",
        description="Compute the unique q-reduced representative of a graph "
        "divisor under the standard connected-graph convention, plus "
        "the exact firing vector f satisfying D_reduced = D - L f.",
        request_type=QReducedRequest,
        result_type=QReducedResult,
        run=compute_q_reduced,
        tags=("graph-theory", "chip-firing", "exact"),
        examples=(
            OperationExample(
                name="triangle_sink_a",
                description="Compute the q-reduced form of a divisor on a triangle "
                "graph with sink at vertex a.",
                input={
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
    MathTool(
        operation_id="graph.chip_firing.canonical_divisor.compute",
        title="Graph canonical divisor",
        description="Compute the graph canonical divisor K(v) = degree(v) - 2 "
        "and its exact degree 2|E| - 2|V|.",
        request_type=CanonicalDivisorRequest,
        result_type=CanonicalDivisorResult,
        run=compute_canonical_divisor,
        tags=("graph-theory", "chip-firing", "exact"),
        examples=(
            OperationExample(
                name="path_graph_3",
                description="Compute the canonical divisor of a path graph on 3 vertices.",
                input={"graph": _GRAPH},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.chip_firing.critical_group.compute",
        title="Critical group (sandpile group)",
        description="Compute the critical group of a connected graph via Smith "
        "normal form of the reduced Laplacian. Returns invariant "
        "factors and group order.",
        request_type=CriticalGroupRequest,
        result_type=CriticalGroupResult,
        run=compute_critical_group,
        tags=("graph-theory", "chip-firing", "exact"),
        examples=(
            OperationExample(
                name="triangle_sink_a",
                description="Compute the critical group of a triangle graph with sink at vertex a.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                    },
                    "sink": "a",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.chip_firing.abel_jacobi.compute",
        title="Abel-Jacobi coordinates",
        description="Map a degree-zero graph divisor into the critical group via "
        "the Abel-Jacobi map, returning canonical class coordinates in "
        "the cokernel of the reduced Laplacian of a connected graph. "
        "Coordinates use row/column Hermite unit reductions and the pinned "
        "SymPy Smith basis, with unit factors omitted. The HNF residual must "
        "fit the conservative transformation work and intermediate-height envelope.",
        request_type=AbelJacobiRequest,
        result_type=AbelJacobiResult,
        run=compute_abel_jacobi,
        tags=("graph-theory", "chip-firing", "exact"),
        examples=(
            OperationExample(
                name="triangle_sink_a",
                description="Map a degree-zero divisor on a triangle graph with "
                "sink at vertex a into the critical group.",
                input={
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
