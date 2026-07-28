"""Exact and explicitly bounded finite simple-graph invariants."""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable
from typing import Any, cast

import networkx as nx
import sympy

from jacobian.contracts.capabilities import (
    CapabilityDiagnostic,
    CapabilityInvocationExample,
)
from jacobian.contracts.graph_invariant_operations import (
    GraphCardinalityMaximumObligation,
    GraphCliqueNumberResult,
    GraphCoreRequest,
    GraphCoreResult,
    GraphDiameterResult,
    GraphEdgeConnectivityResult,
    GraphEulerianResult,
    GraphGirthResult,
    GraphIndependenceNumberResult,
    GraphInvariantRequest,
    GraphMaximumMatchingResult,
    GraphRadiusResult,
    GraphSpanningTreeCountResult,
    GraphTriangleCountResult,
    GraphTutteBergeCertificate,
    GraphVertexConnectivityResult,
)
from jacobian.contracts.graph_optimization import (
    GraphOptimizationRequest,
    OptimizationSearchStep,
    OptimizationTermination,
)
from jacobian.contracts.results import ContractModel
from jacobian.domains._examples import example
from jacobian.domains.graph_optimization.operations import build_simple_graph
from jacobian.operations import (
    BoundedSearchIncomplete,
    BoundedSearchOperation,
    BoundedSearchOutcome,
    BoundedSearchWitness,
    ComputedNotApplicable,
    ComputedOperation,
    ComputedOutcome,
    ComputedSuccess,
)

_INVALID_REQUEST = CapabilityDiagnostic(
    code="INVALID_GRAPH_INVARIANT_REQUEST",
    stage="graph_invariant_input_validation",
    message="Input does not satisfy the bounded finite simple-graph contract.",
    hint="Supply a canonical simple graph with at most 32 vertices.",
)


def _computed[
    ResultT: ContractModel,
](
    capability_id: str,
    title: str,
    description: str,
    result_model: type[ResultT],
    operation: Callable[[nx.Graph[str]], ResultT],
    *tags: str,
    version: str = "1",
    invocation_examples: tuple[CapabilityInvocationExample, ...] = (),
) -> ComputedOperation[GraphInvariantRequest, ResultT]:
    def implementation(
        request: GraphInvariantRequest,
    ) -> ComputedOutcome[ResultT]:
        try:
            graph = cast("nx.Graph[str]", build_simple_graph(request.graph))
            return ComputedSuccess(operation(graph))
        except (ArithmeticError, nx.NetworkXError, TypeError, ValueError) as exc:
            return ComputedNotApplicable(
                CapabilityDiagnostic(
                    code="GRAPH_INVARIANT_NOT_APPLICABLE",
                    stage="graph_invariant_computation",
                    message=str(exc),
                    hint="Check the invariant's graph preconditions.",
                )
            )

    return ComputedOperation(
        capability_id=capability_id,
        title=title,
        description=description,
        request_model=GraphInvariantRequest,
        result_model=result_model,
        implementation=implementation,
        relation_id=capability_id.removesuffix(".compute") + ".relation",
        tags=("graph", "invariant", *tags),
        invalid_request=_INVALID_REQUEST,
        version=version,
        invocation_examples=invocation_examples,
    )


def _girth(graph: nx.Graph[str]) -> GraphGirthResult:
    value = nx.girth(graph)
    girth = 0 if math.isinf(value) else int(value)
    return GraphGirthResult(girth=girth, has_cycle=girth > 0)


def _diameter(graph: nx.Graph[str]) -> GraphDiameterResult:
    if not graph or not nx.is_connected(graph):
        return GraphDiameterResult(
            status="NOT_APPLICABLE",
            connected=False,
            exactness="NOT_APPLICABLE",
            detail="diameter requires a nonempty connected graph",
        )
    return GraphDiameterResult(
        status="COMPUTED",
        diameter=int(nx.diameter(graph)),
        connected=True,
        exactness="EXACT",
    )


def _edge_connectivity(graph: nx.Graph[str]) -> GraphEdgeConnectivityResult:
    value = 0 if len(graph) <= 1 else int(nx.edge_connectivity(graph))
    return GraphEdgeConnectivityResult(edge_connectivity=value)


def _vertex_connectivity(graph: nx.Graph[str]) -> GraphVertexConnectivityResult:
    value = 0 if len(graph) <= 1 else int(nx.node_connectivity(graph))
    return GraphVertexConnectivityResult(vertex_connectivity=value)


def _eulerian(graph: nx.Graph[str]) -> GraphEulerianResult:
    return GraphEulerianResult(is_eulerian=bool(nx.is_eulerian(graph)))


def _spanning_tree_count(graph: nx.Graph[str]) -> GraphSpanningTreeCountResult:
    if not graph:
        return GraphSpanningTreeCountResult(
            spanning_tree_count=0,
            connected=False,
        )
    if not nx.is_connected(graph):
        return GraphSpanningTreeCountResult(
            spanning_tree_count=0,
            connected=False,
        )
    if len(graph) == 1:
        return GraphSpanningTreeCountResult(
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
        spanning_tree_count=int(laplacian[:-1, :-1].det()),
        connected=True,
    )


def _maximum_matching(graph: nx.Graph[str]) -> GraphMaximumMatchingResult:
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
    return GraphMaximumMatchingResult(
        maximum_matching_cardinality=cardinality,
        witness_edges=edges,
        certificate=GraphTutteBergeCertificate(
            barrier_vertices=barrier,
            odd_component_count=odd_component_count,
            upper_bound=cardinality,
        ),
    )


def _triangle_count(graph: nx.Graph[str]) -> GraphTriangleCountResult:
    triangle_counts = cast(dict[str, int], nx.triangles(graph))
    return GraphTriangleCountResult(triangle_count=sum(triangle_counts.values()) // 3)


def _radius(graph: nx.Graph[str]) -> GraphRadiusResult:
    if not graph or not nx.is_connected(graph):
        return GraphRadiusResult(
            status="NOT_APPLICABLE",
            connected=False,
            exactness="NOT_APPLICABLE",
            detail="radius requires a nonempty connected graph",
        )
    return GraphRadiusResult(
        status="COMPUTED",
        radius=int(nx.radius(graph)),
        connected=True,
        exactness="EXACT",
    )


def _k_core_execute(
    request: GraphCoreRequest,
) -> ComputedOutcome[GraphCoreResult]:
    try:
        graph = cast("nx.Graph[str]", build_simple_graph(request.graph))
        core = nx.k_core(graph, k=request.k)
        return ComputedSuccess(
            GraphCoreResult(
                k=request.k,
                vertices=tuple(sorted(str(vertex) for vertex in core.nodes)),
            )
        )
    except (nx.NetworkXError, TypeError, ValueError) as exc:
        return ComputedNotApplicable(
            CapabilityDiagnostic(
                code="GRAPH_INVARIANT_NOT_APPLICABLE",
                stage="graph_invariant_computation",
                message=str(exc),
                hint="Check the invariant's graph preconditions.",
            )
        )


def _maximum_cardinality(
    request: GraphOptimizationRequest,
    *,
    independent: bool,
) -> GraphCliqueNumberResult | GraphIndependenceNumberResult:
    z3: Any = importlib.import_module("z3")
    source = cast("nx.Graph[str]", build_simple_graph(request.graph))
    graph = nx.complement(source) if independent else source
    vertices = tuple(request.graph.vertices)
    result_model = (
        GraphIndependenceNumberResult if independent else GraphCliqueNumberResult
    )
    if not vertices:
        return result_model(
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

    incumbent = tuple(sorted(nx.approximation.max_clique(graph)))
    tested: list[OptimizationSearchStep] = []
    started = time.monotonic()
    termination: OptimizationTermination = "BOUND_CONVERGENCE"
    upper_bound = len(vertices)
    exact = False
    for bound in range(upper_bound, len(incumbent), -1):
        if len(tested) >= request.resource_budget.max_solver_calls:
            termination = "SOLVER_CALL_LIMIT"
            break
        remaining_ms = int(
            (request.resource_budget.wall_seconds - (time.monotonic() - started)) * 1000
        )
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

    return result_model(
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


def _clique_execute(
    request: GraphOptimizationRequest,
) -> BoundedSearchOutcome[GraphCliqueNumberResult]:
    result = cast(
        GraphCliqueNumberResult,
        _maximum_cardinality(request, independent=False),
    )
    if result.status == "EXACT":
        return BoundedSearchWitness(result)
    return BoundedSearchIncomplete(result)


def _independence_execute(
    request: GraphOptimizationRequest,
) -> BoundedSearchOutcome[GraphIndependenceNumberResult]:
    result = cast(
        GraphIndependenceNumberResult,
        _maximum_cardinality(request, independent=True),
    )
    if result.status == "EXACT":
        return BoundedSearchWitness(result)
    return BoundedSearchIncomplete(result)


def _scope(
    request: GraphOptimizationRequest,
    result: ContractModel,
) -> dict[str, object]:
    del result
    return {
        "order": len(request.graph.vertices),
        "wall_seconds": request.resource_budget.wall_seconds,
        "max_solver_calls": request.resource_budget.max_solver_calls,
        "max_order": request.resource_budget.max_order,
    }


def _obligation(
    request: GraphOptimizationRequest,
    result: GraphCliqueNumberResult | GraphIndependenceNumberResult,
    *,
    independent: bool,
) -> GraphCardinalityMaximumObligation:
    return GraphCardinalityMaximumObligation(
        graph=request.graph,
        predicate=(
            "GRAPH_INDEPENDENCE_NUMBER_OPTIMALITY"
            if independent
            else "GRAPH_CLIQUE_NUMBER_OPTIMALITY"
        ),
        status=result.status,
        claimed_value=result.optimum_value,
        lower_bound=result.lower_bound,
        upper_bound=result.upper_bound,
        witness_vertices=result.witness_vertices,
        tested=result.tested,
    )


CLIQUE_NUMBER_CAPABILITY = BoundedSearchOperation(
    capability_id="graph.invariant.clique_number.compute",
    title="Clique number",
    description="Compute a maximum clique under explicit finite search budgets.",
    request_model=GraphOptimizationRequest,
    result_model=GraphCliqueNumberResult,
    implementation=_clique_execute,
    relation_id="graph.invariant.clique_number.relation",
    scope_parameters=_scope,
    is_complete=lambda result: result.status == "EXACT",
    obligation_model=GraphCardinalityMaximumObligation,
    obligation=lambda request, result: _obligation(request, result, independent=False),
    incomplete_basis="the bounded threshold search did not establish optimality",
    tags=("graph", "invariant", "clique", "maximum", "bounded", "z3"),
    invalid_request=_INVALID_REQUEST,
)

INDEPENDENCE_NUMBER_CAPABILITY = BoundedSearchOperation(
    capability_id="graph.invariant.independence_number.compute",
    title="Independence number",
    description=(
        "Compute a maximum independent set under explicit finite search budgets."
    ),
    request_model=GraphOptimizationRequest,
    result_model=GraphIndependenceNumberResult,
    implementation=_independence_execute,
    relation_id="graph.invariant.independence_number.relation",
    scope_parameters=_scope,
    is_complete=lambda result: result.status == "EXACT",
    obligation_model=GraphCardinalityMaximumObligation,
    obligation=lambda request, result: _obligation(request, result, independent=True),
    incomplete_basis="the bounded threshold search did not establish optimality",
    tags=("graph", "invariant", "independent-set", "maximum", "bounded", "z3"),
    invalid_request=_INVALID_REQUEST,
)

BOUNDED_GRAPH_INVARIANT_CAPABILITIES = (
    CLIQUE_NUMBER_CAPABILITY,
    INDEPENDENCE_NUMBER_CAPABILITY,
)

EXACT_GRAPH_INVARIANT_CAPABILITIES = (
    _computed(
        "graph.invariant.triangle_count.compute",
        "Triangle count",
        "Count the three-vertex cycles in a finite simple graph.",
        GraphTriangleCountResult,
        _triangle_count,
        "triangle",
        "exact",
    ),
    _computed(
        "graph.invariant.radius.compute",
        "Graph radius",
        "Compute the minimum eccentricity of a connected graph.",
        GraphRadiusResult,
        _radius,
        "radius",
        "exact",
        version="2",
    ),
    ComputedOperation(
        capability_id="graph.k_core.compute",
        title="Compute a graph k-core",
        description="Return the unique maximal induced subgraph of minimum degree k.",
        request_model=GraphCoreRequest,
        result_model=GraphCoreResult,
        implementation=_k_core_execute,
        relation_id="graph.relation.k-core-of",
        tags=("graph", "invariant", "k-core", "exact"),
        invalid_request=_INVALID_REQUEST,
    ),
    _computed(
        "graph.invariant.girth.compute",
        "Girth",
        "Compute the shortest-cycle length, using zero for an acyclic graph.",
        GraphGirthResult,
        _girth,
        "girth",
        "exact",
    ),
    _computed(
        "graph.invariant.diameter.compute",
        "Diameter",
        "Compute the exact diameter, using -1 for a disconnected graph.",
        GraphDiameterResult,
        _diameter,
        "diameter",
        "exact",
        version="2",
    ),
    _computed(
        "graph.invariant.edge_connectivity.compute",
        "Edge connectivity",
        "Compute the minimum edge-cut cardinality.",
        GraphEdgeConnectivityResult,
        _edge_connectivity,
        "edge-connectivity",
        "exact",
    ),
    _computed(
        "graph.invariant.vertex_connectivity.compute",
        "Vertex connectivity",
        "Compute the minimum vertex-cut cardinality.",
        GraphVertexConnectivityResult,
        _vertex_connectivity,
        "vertex-connectivity",
        "exact",
    ),
    _computed(
        "graph.invariant.is_eulerian.compute",
        "Eulerian predicate",
        "Decide whether the graph has an Eulerian circuit.",
        GraphEulerianResult,
        _eulerian,
        "eulerian",
        "exact",
    ),
    _computed(
        "graph.invariant.spanning_tree_count.compute",
        "Spanning-tree count",
        "Count spanning trees exactly using Kirchhoff's matrix-tree theorem.",
        GraphSpanningTreeCountResult,
        _spanning_tree_count,
        "spanning-tree",
        "exact",
    ),
    _computed(
        "graph.invariant.maximum_matching.compute",
        "Maximum matching",
        "Compute an exact maximum-cardinality matching and its edge witness.",
        GraphMaximumMatchingResult,
        _maximum_matching,
        "matching",
        "maximum",
        "exact",
        version="2",
        invocation_examples=(
            example(
                "triangle_with_tail",
                "Compute and certify a maximum matching of a triangle with one tail.",
                {
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

GRAPH_INVARIANT_CAPABILITIES = (
    *BOUNDED_GRAPH_INVARIANT_CAPABILITIES,
    *EXACT_GRAPH_INVARIANT_CAPABILITIES,
)
