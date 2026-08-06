"""Bounded finite-graph optimization capability tests."""

from __future__ import annotations

import itertools
import shutil
from collections.abc import Iterator
from pathlib import Path

import networkx as nx
import pytest
import z3

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.runtime import create_runtime
from jacobian.runtime.model import JacobianRuntime


@pytest.fixture(scope="module")
def oracle_runtime(
    tmp_path_factory: pytest.TempPathFactory,
    complete_portfolio_template: Path,
) -> Iterator[JacobianRuntime]:
    """Reuse the immutable core store snapshot for shared oracle invokes."""

    root = tmp_path_factory.mktemp("finite-graph-oracles")
    shutil.copytree(complete_portfolio_template, root, dirs_exist_ok=True)
    runtime = create_runtime(root)
    # Pay Z3/solver startup once in fixture setup instead of on the first case.
    warm = nx.relabel_nodes(nx.path_graph(3), lambda vertex: f"v{vertex}")
    _invoke(runtime, "graph.domination.minimum.compute", warm)
    try:
        yield runtime
    finally:
        runtime.close()


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
    runtime: JacobianRuntime,
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


@pytest.mark.parametrize("capability_id", _ORACLE_CAPABILITIES)
@pytest.mark.parametrize(
    "graph",
    (
        nx.path_graph(4),
        nx.cycle_graph(5),
        nx.complete_graph(4),
        nx.disjoint_union(nx.path_graph(3), nx.path_graph(2)),
    ),
    ids=("path", "odd-cycle", "complete", "disconnected"),
)
def test_graph_optimizer_matches_independent_small_brute_force_oracle(
    oracle_runtime: JacobianRuntime,
    capability_id: str,
    graph: nx.Graph[int],
) -> None:
    relabeled: nx.Graph[str] = nx.relabel_nodes(graph, lambda vertex: f"v{vertex}")

    result = _invoke(oracle_runtime, capability_id, relabeled)

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "EXACT"
    assert result.output["optimum_value"] == _brute_force_optimum(
        capability_id, relabeled
    )


@pytest.mark.parametrize(
    ("capability_id", "graph", "optimum", "witness_field", "predicate"),
    (
        (
            "graph.domination.minimum.compute",
            nx.cycle_graph(5, create_using=nx.Graph),
            2,
            "witness_vertices",
            "GRAPH_DOMINATION_MINIMUM_OPTIMALITY",
        ),
        (
            "graph.matching.maximal.minimum.compute",
            nx.cycle_graph(6, create_using=nx.Graph),
            2,
            "witness_edges",
            "GRAPH_MINIMUM_MAXIMAL_MATCHING_OPTIMALITY",
        ),
        (
            "graph.induced_forest.maximum.compute",
            nx.complete_graph(4, create_using=nx.Graph),
            2,
            "witness_vertices",
            "GRAPH_INDUCED_FOREST_MAXIMUM_OPTIMALITY",
        ),
        (
            "graph.induced_tree.maximum.compute",
            nx.cycle_graph(4, create_using=nx.Graph),
            3,
            "witness_vertices",
            "GRAPH_INDUCED_TREE_MAXIMUM_OPTIMALITY",
        ),
        (
            "graph.induced_bipartite.maximum.compute",
            nx.complete_graph(5, create_using=nx.Graph),
            2,
            "witness_vertices",
            "GRAPH_INDUCED_BIPARTITE_MAXIMUM_OPTIMALITY",
        ),
    ),
)
def test_graph_optimizer_returns_exact_witness_and_open_obligation(
    tmp_path: Path,
    capability_id: str,
    graph: nx.Graph[int],
    optimum: int,
    witness_field: str,
    predicate: str,
) -> None:
    relabeled = nx.relabel_nodes(graph, lambda vertex: f"v{vertex}")
    runtime = create_runtime(tmp_path)

    result = _invoke(runtime, capability_id, relabeled)

    assert result.output["status"] == "EXACT"
    assert result.output["optimum_value"] == optimum
    assert result.output["lower_bound"] == optimum
    assert result.output["upper_bound"] == optimum
    assert len(result.output[witness_field]) == optimum
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert len(result.artifact_uris) == 3
    input_uri, result_uri, obligation_uri = result.artifact_uris
    assert runtime.core.store.get(result_uri).manifest.parents == (input_uri,)
    obligation = runtime.core.store.get(obligation_uri)
    assert frozenset(obligation.manifest.parents) == frozenset((input_uri, result_uri))
    assert obligation.payload["predicate"] == predicate
    assert result.obligations[0].obligation_uri == obligation_uri
    if capability_id == "graph.domination.minimum.compute":
        assert nx.is_dominating_set(relabeled, result.output["witness_vertices"])
    elif capability_id == "graph.matching.maximal.minimum.compute":
        matching = {tuple(edge) for edge in result.output["witness_edges"]}
        assert nx.is_matching(relabeled, matching)
        assert nx.is_maximal_matching(relabeled, matching)
    else:
        induced = relabeled.subgraph(result.output["witness_vertices"])
        if capability_id == "graph.induced_forest.maximum.compute":
            assert nx.is_forest(induced)
        elif capability_id == "graph.induced_tree.maximum.compute":
            assert nx.is_tree(induced)
        else:
            assert nx.is_bipartite(induced)


def test_solver_call_budget_preserves_incumbent_without_claiming_optimum(
    tmp_path: Path,
) -> None:
    runtime = create_runtime(tmp_path)
    graph = nx.relabel_nodes(nx.complete_graph(5), lambda vertex: f"v{vertex}")

    result = _invoke(
        runtime,
        "graph.induced_forest.maximum.compute",
        graph,
        max_solver_calls=1,
    )

    assert result.output["status"] == "UNKNOWN"
    assert result.output["optimum_value"] is None
    assert result.output["termination_reason"] == "SOLVER_CALL_LIMIT"
    assert result.output["lower_bound"] == 1
    assert result.output["upper_bound"] == 4
    assert len(result.output["witness_vertices"]) == 1
    assert result.completeness.status is CapabilityCompletenessStatus.UNKNOWN
    obligation = runtime.core.store.get(result.artifact_uris[2])
    assert obligation.payload["claimed_value"] is None


def test_solver_timeout_preserves_partial_witness_as_non_conclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = create_runtime(tmp_path)
    graph = nx.path_graph(["a", "b", "c", "d"])
    monkeypatch.setattr(z3.Solver, "check", lambda _solver: z3.unknown)

    result = _invoke(runtime, "graph.domination.minimum.compute", graph)

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.output["status"] == "UNKNOWN"
    assert result.completeness.status is CapabilityCompletenessStatus.UNKNOWN
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert result.diagnostics[0].code == "GRAPH_OPTIMIZATION_TIMEOUT"
    assert len(result.artifact_uris) == 3


@pytest.mark.parametrize(
    ("solver_name", "capability_id", "update"),
    (
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
    ),
)
def test_invalid_solver_witness_fails_closed_before_artifact_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    solver_name: str,
    capability_id: str,
    update: dict[str, object],
) -> None:
    from jacobian.domains.graph_optimization import finite_optimization

    original = getattr(finite_optimization, solver_name)

    def invalid_witness(*args, **kwargs):
        result = original(*args, **kwargs)
        return result.model_copy(update=update)

    monkeypatch.setattr(finite_optimization, solver_name, invalid_witness)
    runtime = create_runtime(tmp_path)
    result = _invoke(
        runtime,
        capability_id,
        nx.path_graph(["a", "b", "c"]),
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "GRAPH_OPTIMIZATION_WITNESS_INVALID"
    assert result.artifact_uris == ()


@pytest.mark.parametrize(
    "capability_id",
    (
        "graph.domination.minimum.compute",
        "graph.matching.maximal.minimum.compute",
        "graph.induced_forest.maximum.compute",
        "graph.induced_tree.maximum.compute",
        "graph.induced_bipartite.maximum.compute",
    ),
)
def test_empty_graph_boundary_is_exact_zero(
    tmp_path: Path,
    capability_id: str,
) -> None:
    runtime = create_runtime(tmp_path)
    result = _invoke(runtime, capability_id, nx.Graph())

    assert result.output["status"] == "EXACT"
    assert result.output["optimum_value"] == 0
    assert result.output["incumbent_value"] == 0
    assert result.output["termination_reason"] == "SPECIAL_CASE"


def test_order_budget_fails_before_artifact_writes(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
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
