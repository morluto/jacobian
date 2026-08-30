from __future__ import annotations

import pytest

from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.common_neighbor_profile.operations import (
    compute_common_neighbor_profile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices, edges):
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((a, b) for a, b in edges),
    )


def test_c4_fixture() -> None:
    """C4: opposite pairs have codegree 2, adjacent pairs have codegree 0."""
    g = _graph(["0", "1", "2", "3"], [("0", "1"), ("1", "2"), ("2", "3"), ("0", "3")])
    result = compute_common_neighbor_profile(g)
    row_map = {(r.vertex_u, r.vertex_v): r for r in result.rows}
    assert row_map[("0", "2")].codegree == 2
    assert set(row_map[("0", "2")].common_neighbors) == {"1", "3"}
    assert row_map[("0", "1")].codegree == 0
    assert row_map[("1", "3")].codegree == 2
    assert set(row_map[("1", "3")].common_neighbors) == {"0", "2"}


def test_k3() -> None:
    """K3: every pair has one common neighbour (codegree 1)."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")])
    result = compute_common_neighbor_profile(g)
    for row in result.rows:
        assert row.codegree == 1


def test_edgeless_graph() -> None:
    """Edgeless graph has all codegrees 0."""
    g = _graph(["a", "b", "c"], [])
    result = compute_common_neighbor_profile(g)
    for row in result.rows:
        assert row.codegree == 0


def test_row_count() -> None:
    """Number of rows = C(n, 2)."""
    g = _graph(["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("c", "d")])
    result = compute_common_neighbor_profile(g)
    assert len(result.rows) == 6  # C(4,2)


def test_replay_intersection() -> None:
    """Each row equals the intersection of the two source neighbourhoods."""
    g = _graph(["a", "b", "c", "d"], [("a", "b"), ("a", "c"), ("b", "c"), ("b", "d")])
    adjacency: dict[str, set[str]] = {v: set() for v in g.vertices}
    for a, b in g.edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    result = compute_common_neighbor_profile(g)
    for row in result.rows:
        expected = sorted(adjacency[row.vertex_u] & adjacency[row.vertex_v])
        assert list(row.common_neighbors) == expected
        assert row.codegree == len(expected)


def test_c4_free_condition() -> None:
    """A graph is C4-free iff every pair has codegree at most 1."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")])
    result = compute_common_neighbor_profile(g)
    is_c4_free = all(r.codegree <= 1 for r in result.rows)
    assert is_c4_free


def test_complete_pair_coverage() -> None:
    """Every unordered distinct pair appears exactly once."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    result = compute_common_neighbor_profile(g)
    seen = set()
    for row in result.rows:
        pair = (row.vertex_u, row.vertex_v)
        assert pair not in seen
        seen.add(pair)
    assert len(seen) == 3  # C(3,2)


def test_result_preserves_source() -> None:
    g = _graph(["a", "b"], [("a", "b")])
    result = compute_common_neighbor_profile(g)
    assert result.graph == g


def test_native_admission_rejects_complete_profile_over_output_bound() -> None:
    """A dense large graph is rejected before retaining millions of labels."""
    vertices = sorted(str(i) for i in range(256))
    edges = [(vertices[i], vertices[j]) for i in range(256) for j in range(i + 1, 256)]
    g = _graph(vertices, edges)
    with pytest.raises(OperationDomainValidationError, match="output bound"):
        compute_common_neighbor_profile(g)


def test_result_bound_uses_actual_common_neighbor_labels() -> None:
    """A long isolated label does not inflate unrelated common-neighbor rows."""
    short = [f"{i:03d}" for i in range(99)]
    isolated = "x" * 64
    vertices = [*short, isolated]
    edges = [(left, right) for i, left in enumerate(short) for right in short[i + 1 :]]
    result = compute_common_neighbor_profile(_graph(vertices, edges))
    assert len(result.rows) == len(vertices) * (len(vertices) - 1) // 2


def test_result_bound_uses_actual_codegree_digit_width() -> None:
    left = ["a" * 60 + suffix for suffix in ("0", "1")]
    right = ["b" * 60 + f"{index:04x}" for index in range(254)]
    graph = _graph(
        [*left, *right],
        [(left_vertex, right_vertex) for left_vertex in left for right_vertex in right],
    )

    result = compute_common_neighbor_profile(graph)
    assert len(encode_strict_json(result.model_dump(mode="json"))) <= (
        CanonicalLimits().max_output_bytes
    )
