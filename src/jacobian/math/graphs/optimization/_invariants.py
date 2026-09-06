"""Exact and explicitly bounded finite simple-graph invariants."""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.graphs.optimization._budget import remaining_ms as _remaining_ms
from jacobian.math.graphs.optimization._chromatic_kernel import build_simple_graph
from jacobian.math.graphs.optimization._invariant_models import (
    GraphCliqueNumberResult,
    GraphCoreRequest,
    GraphCoreResult,
    GraphDiameterResult,
    GraphEdgeConnectivityResult,
    GraphEulerianResult,
    GraphGirthResult,
    GraphInvariantRequest,
    GraphMaximumMatchingRequest,
    GraphMaximumMatchingResult,
    GraphRadiusResult,
    GraphSpanningTreeCountResult,
    GraphTriangleCountResult,
    GraphTutteBergeCertificate,
    GraphVertexConnectivityResult,
)
from jacobian.math.graphs.optimization._models import (
    GraphOptimizationRequest,
    OptimizationSearchStep,
    OptimizationTermination,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_INVARIANTS_WORKER = Path(__file__).with_name("_invariants_worker.py")
_WORKER_OUTPUT_BYTES = 64 * 1024
_WORKER_ERROR_BYTES = 16_384
_WORKER_ADDRESS_SPACE_BYTES = 1_536 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024


def _computed[
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    result_model: type[ResultT],
    operation: Callable[[Any, SimpleUndirectedGraph], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[GraphInvariantRequest, ResultT]:
    def implementation(
        request: GraphInvariantRequest,
    ) -> ResultT:
        graph = cast(Any, build_simple_graph(request.graph))
        return operation(graph, request.graph)

    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=GraphInvariantRequest,
        result_type=result_model,
        run=implementation,
        tags=("graph", "invariant", *tags),
        examples=examples,
    )


def _girth(graph: Any, source: SimpleUndirectedGraph) -> GraphGirthResult:
    import networkx as nx

    value = nx.girth(graph)
    girth = 0 if math.isinf(value) else int(value)
    return GraphGirthResult(graph=source, girth=girth, has_cycle=girth > 0)


def _diameter(graph: Any, source: SimpleUndirectedGraph) -> GraphDiameterResult:
    import networkx as nx

    from jacobian.math.graphs import diameter

    if not graph or not nx.is_connected(graph):
        return GraphDiameterResult(
            graph=source,
            status="NOT_APPLICABLE",
            connected=False,
            detail="diameter requires a nonempty connected graph",
        )
    return GraphDiameterResult(
        graph=source, status="COMPUTED", diameter=diameter(graph), connected=True
    )


def _edge_connectivity(
    graph: Any, source: SimpleUndirectedGraph
) -> GraphEdgeConnectivityResult:
    import networkx as nx

    value = 0 if len(graph) <= 1 else int(nx.edge_connectivity(graph))
    return GraphEdgeConnectivityResult(graph=source, edge_connectivity=value)


def _vertex_connectivity(
    graph: Any, source: SimpleUndirectedGraph
) -> GraphVertexConnectivityResult:
    import networkx as nx

    value = 0 if len(graph) <= 1 else int(nx.node_connectivity(graph))
    return GraphVertexConnectivityResult(graph=source, vertex_connectivity=value)


def _eulerian(graph: Any, source: SimpleUndirectedGraph) -> GraphEulerianResult:
    from jacobian.math.graphs import is_eulerian

    return GraphEulerianResult(graph=source, is_eulerian=is_eulerian(graph))


def _spanning_tree_count(
    graph: Any, source: SimpleUndirectedGraph
) -> GraphSpanningTreeCountResult:
    import networkx as nx
    import sympy

    if not graph:
        return GraphSpanningTreeCountResult(
            graph=source,
            spanning_tree_count=0,
            connected=False,
        )
    if not nx.is_connected(graph):
        return GraphSpanningTreeCountResult(
            graph=source,
            spanning_tree_count=0,
            connected=False,
        )
    if len(graph) == 1:
        return GraphSpanningTreeCountResult(
            graph=source,
            spanning_tree_count=1,
            connected=True,
        )
    vertices = tuple(graph.nodes)
    index = {vertex: offset for offset, vertex in enumerate(vertices)}
    laplacian = sympy.zeros(len(vertices), len(vertices))
    for left, right in graph.edges:
        left_index = index[left]
        right_index = index[right]
        laplacian[left_index, left_index] += 1
        laplacian[right_index, right_index] += 1
        laplacian[left_index, right_index] -= 1
        laplacian[right_index, left_index] -= 1
    return GraphSpanningTreeCountResult(
        graph=source,
        spanning_tree_count=int(laplacian[:-1, :-1].det()),
        connected=True,
    )


def _maximum_matching(
    graph: Any, source: SimpleUndirectedGraph
) -> GraphMaximumMatchingResult:
    import networkx as nx

    raw = nx.max_weight_matching(graph, maxcardinality=True)
    edges = tuple(
        sorted(
            (str(left), str(right))
            if str(left) < str(right)
            else (str(right), str(left))
            for left, right in raw
        )
    )
    cardinality = len(edges)
    exposed_vertices: set[str] = set()
    for vertex in graph:
        reduced = graph.copy()
        reduced.remove_node(vertex)
        if len(nx.max_weight_matching(reduced, maxcardinality=True)) == cardinality:
            exposed_vertices.add(str(vertex))
    barrier = tuple(
        sorted(
            {
                str(neighbor)
                for vertex in exposed_vertices
                for neighbor in graph.neighbors(vertex)
                if str(neighbor) not in exposed_vertices
            }
        )
    )
    reduced = graph.subgraph(set(graph) - set(barrier))
    odd_component_count = sum(
        len(component) % 2 for component in nx.connected_components(reduced)
    )
    # Establish feasibility and the Tutte-Berge equality at the producing
    # kernel boundary. odd(G-U) was computed once above, not in a validator.
    if (
        any(not graph.has_edge(left, right) for left, right in edges)
        or len({vertex for edge in edges for vertex in edge}) != 2 * cardinality
    ):
        raise RuntimeError("matching backend returned invalid source edges")
    if 2 * cardinality != len(graph) + len(barrier) - odd_component_count:
        raise RuntimeError("matching and Tutte-Berge barrier disagree")
    return GraphMaximumMatchingResult(
        graph=source,
        maximum_matching_cardinality=cardinality,
        witness_edges=edges,
        certificate=GraphTutteBergeCertificate(
            barrier_vertices=barrier,
            odd_component_count=odd_component_count,
            upper_bound=cardinality,
        ),
    )


def _maximum_matching_execute(
    request: GraphMaximumMatchingRequest,
) -> GraphMaximumMatchingResult:
    graph = cast(Any, build_simple_graph(request.graph))
    return _maximum_matching(graph, request.graph)


def _triangle_count(
    graph: Any, source: SimpleUndirectedGraph
) -> GraphTriangleCountResult:
    from jacobian.math.graphs import triangle_count

    return GraphTriangleCountResult(graph=source, triangle_count=triangle_count(graph))


def _radius(graph: Any, source: SimpleUndirectedGraph) -> GraphRadiusResult:
    import networkx as nx

    if not graph or not nx.is_connected(graph):
        return GraphRadiusResult(
            graph=source,
            status="NOT_APPLICABLE",
            connected=False,
            detail="radius requires a nonempty connected graph",
        )
    return GraphRadiusResult(
        graph=source, status="COMPUTED", radius=int(nx.radius(graph)), connected=True
    )


def _k_core_execute(
    request: GraphCoreRequest,
) -> GraphCoreResult:
    import networkx as nx

    graph = cast(Any, build_simple_graph(request.graph))
    core = nx.k_core(graph, k=request.k)
    return GraphCoreResult(
        graph=request.graph,
        k=request.k,
        vertices=tuple(sorted(str(vertex) for vertex in core.nodes)),
    )


def _clique_execute_kernel(
    request: GraphOptimizationRequest,
) -> GraphCliqueNumberResult:
    """Compute the clique number via bounded Z3 threshold search."""

    import z3

    started = time.monotonic()
    graph = cast(Any, build_simple_graph(request.graph))
    vertices = tuple(request.graph.vertices)
    if not vertices:
        return GraphCliqueNumberResult(
            graph=request.graph,
            status="EXACT",
            order=0,
            optimum_value=0,
            incumbent_value=0,
            lower_bound=0,
            upper_bound=0,
            witness_vertices=(),
            tested=(),
            termination_reason="SPECIAL_CASE",
            detail="the empty graph has optimum zero",
        )

    incumbent: tuple[str, ...] = (min(vertices),)
    tested: list[OptimizationSearchStep] = []
    termination: OptimizationTermination = "BOUND_CONVERGENCE"
    upper_bound = len(vertices)
    exact = False
    for bound in range(upper_bound, len(incumbent), -1):
        if len(tested) >= request.resource_budget.max_solver_calls:
            termination = "SOLVER_CALL_LIMIT"
            break
        remaining_ms = _remaining_ms(started, request.resource_budget.wall_seconds)
        if remaining_ms <= 0:
            termination = "WALL_TIME"
            break
        solver = z3.Solver()
        solver.set(timeout=max(1, remaining_ms))
        selected = {
            vertex: z3.Bool(f"selected_{index}")
            for index, vertex in enumerate(vertices)
        }
        for left_index, left in enumerate(vertices):
            for right in vertices[left_index + 1 :]:
                if not graph.has_edge(left, right):
                    solver.add(z3.Or(z3.Not(selected[left]), z3.Not(selected[right])))
        solver.add(
            z3.Sum([z3.If(selected[vertex], 1, 0) for vertex in vertices]) >= bound
        )
        status = solver.check()
        if status == z3.unknown:
            tested.append(
                OptimizationSearchStep(
                    bound=bound,
                    relation="AT_LEAST",
                    status="UNKNOWN",
                )
            )
            termination = "SOLVER_UNKNOWN"
            break
        if status == z3.unsat:
            tested.append(
                OptimizationSearchStep(
                    bound=bound,
                    relation="AT_LEAST",
                    status="UNSATISFIABLE",
                )
            )
            upper_bound = bound - 1
            continue
        tested.append(
            OptimizationSearchStep(
                bound=bound,
                relation="AT_LEAST",
                status="SATISFIABLE",
            )
        )
        model = solver.model()
        incumbent = tuple(
            sorted(
                vertex
                for vertex, variable in selected.items()
                if z3.is_true(model.eval(variable, model_completion=True))
            )
        )
        upper_bound = len(incumbent)
        termination = "OPTIMUM_ESTABLISHED"
        exact = True
        break
    else:
        upper_bound = len(incumbent)
        exact = True

    return GraphCliqueNumberResult(
        graph=request.graph,
        status="EXACT" if exact else "UNKNOWN",
        order=len(vertices),
        optimum_value=len(incumbent) if exact else None,
        incumbent_value=len(incumbent),
        lower_bound=len(incumbent),
        upper_bound=upper_bound,
        witness_vertices=incumbent,
        tested=tuple(tested),
        termination_reason=termination,
        detail="bounded Z3 threshold search seeded by a NetworkX approximation",
    )


def _clique_worker_failure(
    request: GraphOptimizationRequest, detail: str
) -> GraphCliqueNumberResult:
    """Return a source-derived unknown result when the isolated worker fails."""

    vertices = tuple(request.graph.vertices)
    witness = () if not vertices else (min(vertices),)
    return GraphCliqueNumberResult(
        graph=request.graph,
        status="UNKNOWN",
        order=len(vertices),
        optimum_value=None,
        incumbent_value=len(witness),
        lower_bound=len(witness),
        upper_bound=len(vertices),
        witness_vertices=witness,
        tested=(),
        termination_reason="SOLVER_UNKNOWN",
        detail=detail,
    )


def _clique_execute(
    request: GraphOptimizationRequest,
) -> GraphCliqueNumberResult:
    """Run the complete Z3 clique transaction in a bounded owner worker."""

    if len(request.graph.vertices) > request.resource_budget.max_order:
        raise OperationDomainValidationError(
            location=("resource_budget", "max_order"),
            code="graph.clique_number.max_order_budget",
            message="graph order exceeds the declared max_order budget",
        )
    deadline = time.monotonic() + request.resource_budget.wall_seconds
    try:
        with TemporaryDirectory(prefix="jacobian-graph-clique-") as directory:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                return _clique_worker_failure(
                    request, "the clique request expired before worker startup"
                )
            completed = run_bounded_process(
                [sys.executable, str(_INVARIANTS_WORKER)],
                input_bytes=json.dumps(
                    request.model_dump(mode="json"), separators=(",", ":")
                ).encode("utf-8"),
                timeout_seconds=remaining_seconds,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_WORKER_OUTPUT_BYTES,
                stderr_limit=_WORKER_ERROR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(request.resource_budget.wall_seconds)),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=directory,
            )
    except OSError:
        return _clique_worker_failure(
            request, "the bounded clique worker could not be started"
        )
    if (
        completed.timed_out
        or completed.cancelled
        or completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        return _clique_worker_failure(
            request, "the bounded clique worker did not establish an outcome"
        )
    if time.monotonic() >= deadline:
        return _clique_worker_failure(
            request, "the clique request expired before response validation"
        )
    try:
        result = GraphCliqueNumberResult.model_validate(
            json.loads(completed.stdout.decode("utf-8"))
        )
        source_vertices = set(request.graph.vertices)
        source_edges = {tuple(sorted(edge)) for edge in request.graph.edges}
        if (
            result.graph != request.graph
            or result.order != len(request.graph.vertices)
            or not set(result.witness_vertices) <= source_vertices
            or any(
                tuple(sorted(edge)) not in source_edges
                for edge in combinations(result.witness_vertices, 2)
            )
        ):
            raise ValueError("worker result is not bound to the submitted graph")
        if time.monotonic() < deadline:
            return result
        raise ValueError("request expired during response validation")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return _clique_worker_failure(
            request, "the bounded clique worker returned malformed output"
        )


CLIQUE_NUMBER_OPERATION = MathTool(
    operation_id="graph.invariant.clique_number.compute",
    title="Clique number",
    description="Compute a maximum clique under explicit finite search budgets.",
    request_type=GraphOptimizationRequest,
    result_type=GraphCliqueNumberResult,
    run=_clique_execute,
    tags=("graph", "invariant", "clique", "maximum", "bounded", "z3"),
    examples=(
        OperationExample(
            name="path_graph_3",
            description="Clique number of the path graph P3.",
            input={
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"]],
                },
            },
        ),
    ),
)

EXACT_GRAPH_INVARIANT_OPERATIONS = (
    _computed(
        "graph.invariant.triangle_count.compute",
        "Triangle count",
        "Count the three-vertex cycles in a finite simple graph.",
        GraphTriangleCountResult,
        _triangle_count,
        "triangle",
        "exact",
        examples=(
            OperationExample(
                name="triangle_graph",
                description="Count triangles in a three-cycle.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                    }
                },
            ),
        ),
    ),
    _computed(
        "graph.invariant.radius.compute",
        "Graph radius",
        "Compute the minimum eccentricity of a nonempty connected graph.",
        GraphRadiusResult,
        _radius,
        "radius",
        "exact",
        examples=(
            OperationExample(
                name="path_three_radius",
                description="Compute the radius of a three-vertex path.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"]],
                    }
                },
            ),
        ),
    ),
    _computed(
        "graph.invariant.diameter.compute",
        "Graph diameter",
        "Compute the exact diameter of a nonempty connected graph; disconnected graphs return NOT_APPLICABLE.",
        GraphDiameterResult,
        _diameter,
        "diameter",
        "exact",
        examples=(
            OperationExample(
                name="path_three_diameter",
                description="Compute the diameter of a three-vertex path.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"]],
                    }
                },
            ),
        ),
    ),
    _computed(
        "graph.invariant.is_eulerian.compute",
        "Eulerian predicate",
        "Decide whether a finite simple graph has an Eulerian circuit.",
        GraphEulerianResult,
        _eulerian,
        "eulerian",
        "exact",
        examples=(
            OperationExample(
                name="triangle_eulerian",
                description="Decide whether a triangle has an Eulerian circuit.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.k_core.compute",
        title="Compute a graph k-core",
        description="Return the unique maximal induced subgraph of minimum degree k.",
        request_type=GraphCoreRequest,
        result_type=GraphCoreResult,
        run=_k_core_execute,
        tags=("graph", "invariant", "k-core", "exact"),
        examples=(
            OperationExample(
                name="triangle_two_core",
                description="Compute the 2-core of a triangle.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                    },
                    "k": 2,
                },
            ),
        ),
    ),
    _computed(
        "graph.invariant.girth.compute",
        "Girth",
        "Compute the shortest-cycle length, using zero for an acyclic graph.",
        GraphGirthResult,
        _girth,
        "girth",
        "exact",
        examples=(
            OperationExample(
                name="triangle_girth",
                description="Compute the girth of a triangle.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                    }
                },
            ),
        ),
    ),
    _computed(
        "graph.invariant.edge_connectivity.compute",
        "Edge connectivity",
        "Compute the minimum edge-cut cardinality.",
        GraphEdgeConnectivityResult,
        _edge_connectivity,
        "edge-connectivity",
        "exact",
        examples=(
            OperationExample(
                name="triangle_edge_connectivity",
                description="Compute edge connectivity of a triangle.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                    }
                },
            ),
        ),
    ),
    _computed(
        "graph.invariant.vertex_connectivity.compute",
        "Vertex connectivity",
        "Compute the minimum vertex-cut cardinality.",
        GraphVertexConnectivityResult,
        _vertex_connectivity,
        "vertex-connectivity",
        "exact",
        examples=(
            OperationExample(
                name="triangle_vertex_connectivity",
                description="Compute vertex connectivity of a triangle.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                    }
                },
            ),
        ),
    ),
    _computed(
        "graph.invariant.spanning_tree_count.compute",
        "Spanning-tree count",
        "Count spanning trees exactly using Kirchhoff's matrix-tree theorem.",
        GraphSpanningTreeCountResult,
        _spanning_tree_count,
        "spanning-tree",
        "exact",
        examples=(
            OperationExample(
                name="triangle_spanning_trees",
                description="Count spanning trees of a triangle.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.invariant.maximum_matching.compute",
        title="Maximum matching",
        description=(
            "Compute an exact maximum-cardinality matching and a Tutte-Berge "
            "upper-bound certificate for one simple graph of at most 256 vertices."
        ),
        request_type=GraphMaximumMatchingRequest,
        result_type=GraphMaximumMatchingResult,
        run=_maximum_matching_execute,
        tags=("graph", "invariant", "matching", "maximum", "exact"),
        examples=(
            OperationExample(
                name="triangle_with_tail",
                description="Compute and certify a maximum matching of a triangle with one tail.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c", "d"],
                        "edges": [
                            ["a", "b"],
                            ["a", "c"],
                            ["b", "c"],
                            ["c", "d"],
                        ],
                    }
                },
            ),
        ),
    ),
)
