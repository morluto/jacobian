"""NetworkX/Z3 finite-graph optimization capability contracts."""

from __future__ import annotations

import itertools

import networkx as nx
import pytest
import z3
from tests.support.services import DomainTestServices

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus


def _payload(graph: nx.Graph[str], **budget: int) -> dict[str, object]:
    return {
        "graph": {
            "vertices": sorted(graph.nodes),
            "edges": [sorted(edge) for edge in graph.edges],
        },
        "resource_budget": {
            "wall_seconds": 5,
            "max_solver_calls": 33,
            "max_order": 32,
            **budget,
        },
    }


def _invoke(
    runtime: DomainTestServices,
    capability_id: str,
    graph: nx.Graph[str],
    **budget: int,
):
    return runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=capability_id,
            input=_payload(graph, **budget),
        )
    )


def _vertex_subsets(graph: nx.Graph[str]):
    vertices = tuple(graph.nodes)
    return (
        subset
        for size in range(len(vertices) + 1)
        for subset in itertools.combinations(vertices, size)
    )


def _edge_subsets(graph: nx.Graph[str]):
    edges = tuple(graph.edges)
    return (
        subset
        for size in range(len(edges) + 1)
        for subset in itertools.combinations(edges, size)
    )


def _brute_force_optimum(capability_id: str, graph: nx.Graph[str]) -> int:
    assert len(graph) <= 6, (
        f"brute-force oracle is exponential; got {len(graph)} vertices (max 6)"
    )
    assert graph.number_of_edges() <= 10, (
        "brute-force oracle is exponential; "
        f"got {graph.number_of_edges()} edges (max 10)"
    )
    if capability_id == "graph.domination.minimum.compute":
        return min(
            len(subset)
            for subset in _vertex_subsets(graph)
            if nx.is_dominating_set(graph, subset)
        )
    if capability_id == "graph.matching.maximal.minimum.compute":
        return min(
            len(subset)
            for subset in _edge_subsets(graph)
            if nx.is_matching(graph, subset) and nx.is_maximal_matching(graph, subset)
        )

    if capability_id == "graph.induced_forest.maximum.compute":
        predicate = nx.is_forest
    elif capability_id == "graph.induced_tree.maximum.compute":
        predicate = nx.is_tree
    elif capability_id == "graph.induced_bipartite.maximum.compute":
        predicate = nx.is_bipartite
    else:  # pragma: no cover - protects the test helper from silent extension
        raise AssertionError(f"unsupported capability: {capability_id}")
    return max(
        len(subset)
        for subset in _vertex_subsets(graph)
        if subset and predicate(graph.subgraph(subset))
    )


_ORACLE_CAPABILITIES = (
    "graph.domination.minimum.compute",
    "graph.matching.maximal.minimum.compute",
    "graph.induced_forest.maximum.compute",
    "graph.induced_tree.maximum.compute",
    "graph.induced_bipartite.maximum.compute",
)

_ORACLE_GRAPHS = (
    ("path", nx.path_graph(4)),
    ("odd-cycle", nx.cycle_graph(5)),
    ("complete", nx.complete_graph(4)),
    ("disconnected", nx.disjoint_union(nx.path_graph(3), nx.path_graph(2))),
)


def test_graph_optimizer_matches_independent_small_brute_force_oracle(
    graph_optimization_services: DomainTestServices,
) -> None:
    for graph_name, graph in _ORACLE_GRAPHS:
        for capability_id in _ORACLE_CAPABILITIES:
            case = f"{graph_name}:{capability_id}"
            relabeled: nx.Graph[str] = nx.relabel_nodes(
                graph, lambda vertex: f"v{vertex}"
            )

            result = _invoke(graph_optimization_services, capability_id, relabeled)

            assert result.execution.status is ExecutionStatus.COMPLETED, case
            output = result.output["result"]
            assert output["status"] == "EXACT", case
            assert output["optimum_value"] == _brute_force_optimum(
                capability_id, relabeled
            ), case


_WITNESS_CASES = (
    (
        "graph.domination.minimum.compute",
        nx.cycle_graph(5, create_using=nx.Graph),
        2,
        "witness_vertices",
    ),
    (
        "graph.matching.maximal.minimum.compute",
        nx.cycle_graph(6, create_using=nx.Graph),
        2,
        "witness_edges",
    ),
    (
        "graph.induced_forest.maximum.compute",
        nx.complete_graph(4, create_using=nx.Graph),
        2,
        "witness_vertices",
    ),
    (
        "graph.induced_tree.maximum.compute",
        nx.cycle_graph(4, create_using=nx.Graph),
        3,
        "witness_vertices",
    ),
    (
        "graph.induced_bipartite.maximum.compute",
        nx.complete_graph(5, create_using=nx.Graph),
        2,
        "witness_vertices",
    ),
)


def test_graph_optimizer_returns_exact_typed_witness(
    graph_optimization_services: DomainTestServices,
) -> None:
    runtime = graph_optimization_services

    for capability_id, graph, optimum, witness_field in _WITNESS_CASES:
        case = capability_id
        relabeled = nx.relabel_nodes(graph, lambda vertex: f"v{vertex}")
        result = _invoke(runtime, capability_id, relabeled)

        output = result.output["result"]
        assert output["status"] == "EXACT", case
        assert output["optimum_value"] == optimum, case
        assert output["lower_bound"] == optimum, case
        assert output["upper_bound"] == optimum, case
        assert len(output[witness_field]) == optimum, case
        assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED, case
        assert result.artifact_uris == (), case
        assert result.obligations == (), case
        if capability_id == "graph.domination.minimum.compute":
            assert nx.is_dominating_set(relabeled, output["witness_vertices"]), case
        elif capability_id == "graph.matching.maximal.minimum.compute":
            matching = {tuple(edge) for edge in output["witness_edges"]}
            assert nx.is_matching(relabeled, matching), case
            assert nx.is_maximal_matching(relabeled, matching), case
        else:
            induced = relabeled.subgraph(output["witness_vertices"])
            if capability_id == "graph.induced_forest.maximum.compute":
                assert nx.is_forest(induced), case
            elif capability_id == "graph.induced_tree.maximum.compute":
                assert nx.is_tree(induced), case
            else:
                assert nx.is_bipartite(induced), case


def test_solver_call_budget_preserves_incumbent_without_claiming_optimum(
    graph_optimization_services: DomainTestServices,
) -> None:
    runtime = graph_optimization_services
    graph = nx.relabel_nodes(nx.complete_graph(5), lambda vertex: f"v{vertex}")

    result = _invoke(
        runtime,
        "graph.induced_forest.maximum.compute",
        graph,
        max_solver_calls=1,
    )

    output = result.output["result"]
    assert output["status"] == "UNKNOWN"
    assert output["optimum_value"] is None
    assert output["termination_reason"] == "SOLVER_CALL_LIMIT"
    assert output["lower_bound"] == 1
    assert output["upper_bound"] == 4
    assert len(output["witness_vertices"]) == 1
    assert result.artifact_uris == ()


def test_solver_timeout_is_artifact_free_non_conclusion(
    graph_optimization_services: DomainTestServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = graph_optimization_services
    graph = nx.path_graph(["a", "b", "c", "d"])
    monkeypatch.setattr(z3.Solver, "check", lambda _solver: z3.unknown)

    result = _invoke(runtime, "graph.domination.minimum.compute", graph)

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.output["error"]["code"] == "GRAPH_OPTIMIZATION_TIMEOUT"
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert result.diagnostics[0].code == "GRAPH_OPTIMIZATION_TIMEOUT"
    assert result.artifact_uris == ()


_INVALID_WITNESS_CASES = (
    (
        "solve_domination",
        "graph.domination.minimum.compute",
        {"witness_vertices": ("missing",)},
    ),
    (
        "solve_minimum_maximal_matching",
        "graph.matching.maximal.minimum.compute",
        {"witness_edges": (("a", "missing"),)},
    ),
    (
        "solve_induced_forest",
        "graph.induced_forest.maximum.compute",
        {"witness_vertices": ("missing",)},
    ),
    (
        "solve_induced_tree",
        "graph.induced_tree.maximum.compute",
        {"witness_vertices": ("missing",)},
    ),
    (
        "solve_induced_bipartite",
        "graph.induced_bipartite.maximum.compute",
        {"witness_vertices": ("missing",)},
    ),
)


def test_invalid_solver_witness_fails_closed_before_artifact_writes(
    graph_optimization_services: DomainTestServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.domains.graph_optimization import finite_optimization

    runtime = graph_optimization_services
    for solver_name, capability_id, update in _INVALID_WITNESS_CASES:
        original = getattr(finite_optimization, solver_name)

        def invalid_witness(*args, _original=original, _update=update, **kwargs):
            result = _original(*args, **kwargs)
            return result.model_copy(update=_update)

        with monkeypatch.context() as case_patch:
            case_patch.setattr(finite_optimization, solver_name, invalid_witness)
            result = _invoke(
                runtime,
                capability_id,
                nx.path_graph(["a", "b", "c"]),
            )

        assert result.execution.status is ExecutionStatus.ERROR, capability_id
        assert result.diagnostics[0].code == "GRAPH_OPTIMIZATION_WITNESS_INVALID", (
            capability_id
        )
        assert result.artifact_uris == (), capability_id


def test_empty_graph_boundary_is_exact_zero(
    graph_optimization_services: DomainTestServices,
) -> None:
    runtime = graph_optimization_services
    for capability_id in _ORACLE_CAPABILITIES:
        result = _invoke(runtime, capability_id, nx.Graph())

        output = result.output["result"]
        assert output["status"] == "EXACT", capability_id
        assert output["optimum_value"] == 0, capability_id
        assert output["incumbent_value"] == 0, capability_id
        assert output["termination_reason"] == "SPECIAL_CASE", capability_id


def test_order_budget_fails_before_artifact_writes(
    graph_optimization_services: DomainTestServices,
) -> None:
    runtime = graph_optimization_services
    graph = nx.relabel_nodes(nx.path_graph(3), lambda vertex: f"v{vertex}")

    result = _invoke(
        runtime,
        "graph.domination.minimum.compute",
        graph,
        max_order=2,
    )

    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == "INVALID_GRAPH_OPTIMIZATION_REQUEST"
    assert result.artifact_uris == ()
