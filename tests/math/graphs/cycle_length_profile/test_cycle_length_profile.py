from __future__ import annotations

from jacobian.math.graphs.cycle_length_profile.operations import (
    compute_cycle_length_profile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices, edges):
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((a, b) for a, b in edges),
    )


def test_c4() -> None:
    """C4 has cycle-length spectrum {4}."""
    g = _graph(["0", "1", "2", "3"], [("0", "1"), ("1", "2"), ("2", "3"), ("0", "3")])
    result = compute_cycle_length_profile(g)
    lengths = [r.cycle_length for r in result.rows]
    assert lengths == [4]


def test_k4() -> None:
    """K4 has cycle-length spectrum {3, 4}."""
    g = _graph(
        ["0", "1", "2", "3"],
        [("0", "1"), ("0", "2"), ("0", "3"), ("1", "2"), ("1", "3"), ("2", "3")],
    )
    result = compute_cycle_length_profile(g)
    lengths = [r.cycle_length for r in result.rows]
    assert 3 in lengths
    assert 4 in lengths


def test_triangle() -> None:
    """Triangle has cycle-length spectrum {3}."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")])
    result = compute_cycle_length_profile(g)
    lengths = [r.cycle_length for r in result.rows]
    assert lengths == [3]


def test_path_no_cycles() -> None:
    """Path graph has no cycles."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    result = compute_cycle_length_profile(g)
    assert len(result.rows) == 0


def test_witness_replay() -> None:
    """Each witness is a valid cycle."""
    g = _graph(
        ["0", "1", "2", "3"],
        [("0", "1"), ("1", "2"), ("2", "3"), ("0", "3")],
    )
    result = compute_cycle_length_profile(g)
    edges = set(g.edges)
    for row in result.rows:
        witness = [*list(row.witness), row.witness[0]]
        for i in range(len(witness) - 1):
            a, b = witness[i], witness[i + 1]
            assert (a, b) in edges or (b, a) in edges
        assert row.cycle_length == len(row.witness)


def test_disconnected_components() -> None:
    """Disconnected components are handled."""
    g = _graph(
        ["a", "b", "c", "d"],
        [("a", "b"), ("a", "c"), ("b", "c")],
    )
    result = compute_cycle_length_profile(g)
    lengths = [r.cycle_length for r in result.rows]
    assert lengths == [3]


def test_result_preserves_source() -> None:
    """Result retains the source graph."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")])
    result = compute_cycle_length_profile(g)
    assert result.graph == g


def test_cycle_traversal_allows_descending_indices() -> None:
    """A cycle is found even when its path must go down the vertex axis."""
    g = _graph(
        ["0", "1", "2", "3"],
        [("0", "2"), ("1", "2"), ("1", "3"), ("0", "3")],
    )
    result = compute_cycle_length_profile(g)
    assert [row.cycle_length for row in result.rows] == [4]


def test_cycle_witness_canonicalizes_reverse_orientation() -> None:
    """Canonical witnesses compare both rotations and orientations."""
    g = _graph(["z", "b", "a"], [("b", "z"), ("a", "z"), ("a", "b")])
    result = compute_cycle_length_profile(g)
    assert result.rows[0].witness == ("a", "b", "z")


def test_cycle_witness_preserves_canonical_graph_labels() -> None:
    """Witnesses accept the graph value's complete string-label domain."""
    long_label = "label-" + "x" * 65
    g = _graph(
        ["", "a", long_label],
        [("", "a"), ("", long_label), ("a", long_label)],
    )
    result = compute_cycle_length_profile(g)
    assert result.rows[0].witness == ("", "a", long_label)


def test_sparse_star_is_admitted_without_global_degree_restriction() -> None:
    """A sparse high-degree graph does not inherit a dense-graph estimate."""
    center = ""
    leaves = [str(index) for index in range(1, 16)]
    g = _graph([center, *leaves], [(center, leaf) for leaf in leaves])
    assert compute_cycle_length_profile(g).rows == ()


def test_native_admission_accepts_large_sparse_graph() -> None:
    """Sparse graphs beyond the old cap remain within the derived envelope."""
    graph = _graph([str(i) for i in range(17)], [])
    assert compute_cycle_length_profile(graph).rows == ()
