from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.coloring._models import EdgeColoringAssignment
from jacobian.math.graphs.edge_coloring_arrowing.operations import (
    decide_edge_coloring_arrowing,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices, edges):
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((a, b) for a, b in edges),
    )


def _k3():
    return _graph(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")])


def _k6():
    vs = [str(i) for i in range(6)]
    edges = []
    for i in range(6):
        for j in range(i + 1, 6):
            edges.append((str(i), str(j)))
    return _graph(vs, edges)


def _k5():
    vs = [str(i) for i in range(5)]
    edges = []
    for i in range(5):
        for j in range(i + 1, 5):
            edges.append((str(i), str(j)))
    return _graph(vs, edges)


def test_k6_arrows_k3_k3() -> None:
    """K6 arrows (K3,K3) under red/blue edge colourings."""
    result = decide_edge_coloring_arrowing(_k6(), (_k3(), _k3()))
    assert result.outcome == "ARROWS"


def test_k5_does_not_arrow_k3_k3() -> None:
    """K5 does not arrow (K3,K3): the 5-cycle colouring avoids triangles."""
    result = decide_edge_coloring_arrowing(_k5(), (_k3(), _k3()))
    assert result.outcome == "DOES_NOT_ARROW"
    assert result.avoiding_coloring is not None


def test_k2_arrows_k1_k1() -> None:
    """K2 arrows (K1,K1): any single-edge graph must be one colour."""
    k1_a = _graph(["x"], [])
    k1_b = _graph(["y"], [])
    k2 = _graph(["0", "1"], [("0", "1")])
    result = decide_edge_coloring_arrowing(k2, (k1_a, k1_b))
    assert result.outcome == "ARROWS"


def test_empty_host_does_not_arrow() -> None:
    """Empty host cannot force any target."""
    empty = _graph([], [])
    result = decide_edge_coloring_arrowing(empty, (_k3(), _k3()))
    assert result.outcome == "DOES_NOT_ARROW"


def test_sparse_large_host_uses_derived_work_bound() -> None:
    host = _graph([f"v{i}" for i in range(100)], [("v0", "v1")])
    target = _k3()

    result = decide_edge_coloring_arrowing(host, (target, target))

    assert result.outcome == "DOES_NOT_ARROW"


def test_avoiding_coloring_replay() -> None:
    """Replay the avoiding colouring to verify it avoids all targets."""
    result = decide_edge_coloring_arrowing(_k5(), (_k3(), _k3()))
    assert result.outcome == "DOES_NOT_ARROW"
    assert isinstance(result.avoiding_coloring, EdgeColoringAssignment)
    coloring = dict(enumerate(result.avoiding_coloring.coloring))
    edges = list(result.host_graph.edges)
    for color, target in enumerate(result.targets):
        color_edges = [edges[i] for i in range(len(edges)) if coloring[i] == color]
        host_v = result.host_graph.vertices
        from itertools import permutations

        for va in permutations(host_v, len(target.vertices)):
            vmap = dict(zip(target.vertices, va, strict=True))
            found = True
            for a, b in target.edges:
                ha, hb = vmap[a], vmap[b]
                edge = (min(ha, hb), max(ha, hb))
                if edge not in set(color_edges):
                    found = False
                    break
            assert not found


def test_result_preserves_inputs() -> None:
    """Result retains the original host and targets."""
    k6 = _k6()
    k3 = _k3()
    result = decide_edge_coloring_arrowing(k6, (k3, k3))
    assert result.host_graph == k6
    assert result.targets == (k3, k3)


def test_rejects_empty_target_and_unbounded_search() -> None:
    host = _k6()
    empty = _graph([], [])
    with pytest.raises(OperationDomainValidationError, match="target 0"):
        decide_edge_coloring_arrowing(host, (empty,))
    with pytest.raises(OperationDomainValidationError, match="embedding checks"):
        decide_edge_coloring_arrowing(
            _graph([str(i) for i in range(10)], _k6().edges), (_k3(),) * 6
        )
