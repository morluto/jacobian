from __future__ import annotations

from itertools import combinations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.cycle_length_profile import (
    CycleLengthProfileResult,
    CycleLengthRow,
)
from jacobian.math.graphs.cycle_length_profile.operations import (
    compute_cycle_length_profile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices, edges):
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((a, b) for a, b in edges),
    )


def test_result_types_are_publicly_exported() -> None:
    """Native callers can import the canonical producer result types."""
    result = compute_cycle_length_profile(_graph(["a", "b", "c"], []))
    assert isinstance(result, CycleLengthProfileResult)
    assert result.rows == ()
    assert CycleLengthRow.model_fields["cycle_length"].annotation is not None


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


def test_edgeless_graph_uses_triangular_root_scan_bound() -> None:
    """Root scans for an edgeless graph are charged once per unordered pair."""
    graph = _graph([str(i) for i in range(256)], [])
    assert compute_cycle_length_profile(graph).rows == ()


def test_near_complete_graph_uses_exhaustive_work_bound() -> None:
    """Missing a few edges must not trigger an unproved complete-graph shortcut."""
    vertices = [f"v{index:02d}" for index in range(20)]
    edges = list(combinations(vertices, 2))
    edges.remove(("v00", "v01"))
    edges.remove(("v00", "v02"))
    with pytest.raises(OperationDomainValidationError, match="work bound"):
        compute_cycle_length_profile(_graph(vertices, edges))


def test_wheel_uses_first_witness_bound() -> None:
    """A pancyclic wheel is admitted through its first-witness envelope."""
    vertices = [f"v{index:02d}" for index in range(17)]
    edges = [tuple(sorted((vertices[0], vertices[index]))) for index in range(1, 17)]
    edges.extend(
        tuple(sorted((vertices[index], vertices[1 + (index % 16)])))
        for index in range(1, 17)
    )
    result = compute_cycle_length_profile(_graph(vertices, sorted(edges)))
    assert [row.cycle_length for row in result.rows] == list(range(3, 18))


def test_result_budget_charges_cycle_blocks_independently() -> None:
    """Large labels in one cycle block do not inflate witnesses in another."""
    wheel = [f"w{index:02d}" for index in range(10)]
    triangle = ["t0" + "x" * 500_000, "t1" + "x" * 500_000, "t2" + "x" * 500_000]
    vertices = [*wheel, *triangle]
    rim = wheel[1:]
    edges = [tuple(sorted((rim[index], rim[(index + 1) % 9]))) for index in range(9)]
    edges.extend(tuple(sorted((wheel[0], wheel[index]))) for index in range(1, 10))
    edges.extend(
        [
            (triangle[0], triangle[1]),
            (triangle[0], triangle[2]),
            (triangle[1], triangle[2]),
        ]
    )
    result = compute_cycle_length_profile(_graph(vertices, edges))
    assert {row.cycle_length for row in result.rows} == set(range(3, 11))


def test_simple_cycle_reserves_only_its_feasible_length() -> None:
    """A chordless cycle has one possible witness length, not every prefix."""
    vertices = sorted(f"v{index:02d}-" + "x" * 100_000 for index in range(20))
    edges = [
        tuple(sorted((vertices[index], vertices[(index + 1) % 20])))
        for index in range(20)
    ]
    result = compute_cycle_length_profile(_graph(vertices, sorted(edges)))
    assert [row.cycle_length for row in result.rows] == [20]


def test_bipartite_block_reserves_only_part_size_feasible_lengths() -> None:
    left = [f"l{index}" for index in range(2)]
    right = [f"r{index}" for index in range(8)]
    vertices = [*left, *right]
    edges = [(a, b) for a in left for b in right]

    result = compute_cycle_length_profile(_graph(vertices, edges))

    assert [row.cycle_length for row in result.rows] == [4]


def test_perfect_matching_depth_two_scans_are_charged() -> None:
    vertices = tuple(f"v{index:03d}" for index in range(256))
    graph = _graph(
        vertices,
        [(f"v{index:03d}", f"v{index + 128:03d}") for index in range(128)],
    )

    with pytest.raises(OperationDomainValidationError, match="work bound"):
        compute_cycle_length_profile(graph)


def test_theta_reserves_only_its_two_feasible_lengths() -> None:
    """A theta non-bipartite block admits only its realizable 4- and 5-cycles.

    Three internally disjoint paths of lengths 2, 2, and 3 between two branch
    vertices form a six-vertex non-bipartite block whose only simple cycles have
    lengths 4 and 5.  The output reservation must not charge the impossible
    lengths 3 and 6, or a valid request near the canonical bound is misrejected.
    """
    vertices = ["t0", "t1", "t2", "t3", "t4", "t5"]
    edges = [
        ("t0", "t1"),
        ("t1", "t5"),
        ("t0", "t2"),
        ("t2", "t5"),
        ("t0", "t3"),
        ("t3", "t4"),
        ("t4", "t5"),
    ]
    result = compute_cycle_length_profile(_graph(vertices, edges))
    assert [row.cycle_length for row in result.rows] == [4, 5]


def test_theta_graph_large_labels_stay_within_result_envelope() -> None:
    """A large-label theta graph is admitted when only feasible rows are budgeted.

    Reserving the full 3..6 range for a theta whose only simple cycles have
    lengths 4 and 5 misrejects a request whose exact output fits the 10 MiB
    canonical envelope.
    """
    label = "x" * 300_000
    vertices = [f"t{index}" + label for index in range(6)]
    edges = [
        (vertices[0], vertices[1]),
        (vertices[1], vertices[5]),
        (vertices[0], vertices[2]),
        (vertices[2], vertices[5]),
        (vertices[0], vertices[3]),
        (vertices[3], vertices[4]),
        (vertices[4], vertices[5]),
    ]
    result = compute_cycle_length_profile(_graph(vertices, edges))
    assert {row.cycle_length for row in result.rows} == {4, 5}


def test_wheel_order_is_carried_from_single_recognition() -> None:
    """A wheel admitted once yields every cycle length for either rim order.

    The hub-then-cyclic-rim order is derived once during admission and carried
    into the kernel plan, so a wheel whose rim vertices follow a non-sequential
    axis still exposes its full pancyclic profile.
    """
    rim = [f"rim-{index + 1:02d}" for index in range(15, 0, -1)]
    hub = "hub"
    vertices = [hub, *rim]
    edges = [tuple(sorted((hub, rim_vertex))) for rim_vertex in rim]
    edges.extend(
        tuple(sorted((rim[index], rim[(index + 1) % len(rim)])))
        for index in range(len(rim))
    )
    result = compute_cycle_length_profile(_graph(vertices, edges))
    assert [row.cycle_length for row in result.rows] == list(range(3, 17))
