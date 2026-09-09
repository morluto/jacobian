from __future__ import annotations

import pytest

from jacobian._execution import (
    OperationExecutionTimeoutError,
    OperationResourceExhaustedError,
    OperationWorkLedger,
    bind_request_deadline,
    request_execution,
)
from jacobian.math.graphs.regular_subgraph._models import RegularSubgraphResult
from jacobian.math.graphs.regular_subgraph.operations import (
    find_k_regular_subgraph,
    verify_k_regular_subgraph,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices: list[str], edges: list[tuple[str, str]]) -> SimpleUndirectedGraph:
    canonical_edges = tuple(
        (left, right) if left <= right else (right, left) for left, right in edges
    )
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=canonical_edges,
    )


def test_c4_is_2_regular() -> None:
    """C4 contains a 2-regular subgraph (itself)."""
    g = _graph(["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")])
    result = find_k_regular_subgraph(g, 2)
    assert result.found
    assert len(result.vertices) == 4
    # Every used vertex must have degree 2.
    degree: dict[str, int] = {}
    for u, v in result.edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    assert all(d == 2 for d in degree.values())


def test_p3_no_2_regular_subgraph() -> None:
    """P3 has no nonempty 2-regular subgraph."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    result = find_k_regular_subgraph(g, 2)
    assert not result.found
    assert result.vertices == ()
    assert result.edges == ()


def test_k0_returns_single_vertex() -> None:
    """k=0 returns a single vertex with no edges."""
    g = _graph(["a", "b"], [("a", "b")])
    result = find_k_regular_subgraph(g, 0)
    assert result.found
    assert len(result.vertices) == 1
    assert len(result.edges) == 0


def test_k1_matching() -> None:
    """k=1 subgraph is a matching: any edge."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    result = find_k_regular_subgraph(g, 1)
    assert result.found
    assert len(result.edges) == 1
    degree: dict[str, int] = {}
    for u, v in result.edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    assert all(d == 1 for d in degree.values())


def test_triangle_is_2_regular() -> None:
    """K3 is 2-regular."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")])
    result = find_k_regular_subgraph(g, 2)
    assert result.found
    assert len(result.vertices) == 3
    degree: dict[str, int] = {}
    for u, v in result.edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    assert all(d == 2 for d in degree.values())


def test_k3_is_3_regular_in_k4() -> None:
    """K4 contains a K4 (3-regular subgraph)."""
    g = _graph(
        ["a", "b", "c", "d"],
        [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")],
    )
    result = find_k_regular_subgraph(g, 3)
    assert result.found
    assert len(result.vertices) == 4
    degree: dict[str, int] = {}
    for u, v in result.edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    assert all(d == 3 for d in degree.values())


def test_serialized_witness_claim_is_verified_against_its_graph() -> None:
    graph = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    result = find_k_regular_subgraph(graph, 1)
    assert verify_k_regular_subgraph(
        RegularSubgraphResult.model_validate_json(result.model_dump_json())
    )
    forged = result.model_dump(mode="json")
    forged["edges"] = [["a", "b"], ["b", "c"]]
    assert not verify_k_regular_subgraph(RegularSubgraphResult.model_validate(forged))


def test_widened_path_uses_exact_linear_k2_regime() -> None:
    vertices = tuple(f"v{index:03d}" for index in range(257))
    edges = [(vertices[index], vertices[index + 1]) for index in range(256)]
    graph = _graph(list(vertices), edges)
    result = find_k_regular_subgraph(graph, 2)
    assert not result.found
    assert verify_k_regular_subgraph(result)


def test_k0_and_k1_on_a_long_path_do_not_require_edge_subset_search() -> None:
    vertices = [f"v{index}" for index in range(18)]
    edges = [(vertices[index], vertices[index + 1]) for index in range(17)]
    graph = _graph(vertices, edges)
    zero = find_k_regular_subgraph(graph, 0)
    assert zero.found
    assert zero.vertices == (vertices[0],)
    assert zero.edges == ()
    matching = find_k_regular_subgraph(graph, 1)
    assert matching.found
    assert matching.edges == ((vertices[0], vertices[1]),)


def test_triangle_witness_precedes_oversized_complete_search() -> None:
    vertices = [f"v{index:02d}" for index in range(18)]
    edges = [("v00", "v01"), ("v00", "v02"), ("v01", "v02")]
    edges.extend((vertices[index], vertices[index + 1]) for index in range(3, 17))
    result = find_k_regular_subgraph(_graph(vertices, edges), 2)
    assert result.found
    assert result.vertices == ("v00", "v01", "v02")
    assert result.edges == (("v00", "v01"), ("v00", "v02"), ("v01", "v02"))


def test_dense_large_subsets_charge_their_selected_edge_work() -> None:
    left = [f"l{index:02d}" for index in range(16)]
    right = [f"r{index:02d}" for index in range(16)]
    graph = _graph(left + right, [(u, v) for u in left for v in right])
    with pytest.raises(OperationResourceExhaustedError, match="work allowance"):
        find_k_regular_subgraph(graph, 16)


def test_search_checks_parent_deadline_after_final_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    original_charge = OperationWorkLedger.charge

    def charge_and_expire(ledger: OperationWorkLedger, units: int = 1) -> None:
        original_charge(ledger, units)
        clock["now"] = 2.0

    monkeypatch.setattr(OperationWorkLedger, "charge", charge_and_expire)
    monkeypatch.setattr("jacobian._execution.time.monotonic", lambda: clock["now"])
    graph = _graph(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")])
    with request_execution(0.0):
        bind_request_deadline(1.0)
        with pytest.raises(OperationExecutionTimeoutError):
            find_k_regular_subgraph(graph, 2)
