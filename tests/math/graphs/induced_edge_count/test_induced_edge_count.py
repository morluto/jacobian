from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.induced_edge_count._models import (
    InducedEdgeCountProfileRequest,
)
from jacobian.math.graphs.induced_edge_count.operations import (
    MAX_RETAINED_LABEL_CHARACTERS,
    compute_induced_edge_count_profile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(
    vertices: Sequence[str], edges: Sequence[tuple[str, str]]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((a, b) for a, b in edges),
    )


def test_path3_k2_fixture() -> None:
    """P3 at k=2: subsets {0,1} and {1,2} have 1 edge, {0,2} has 0."""
    g = _graph(["0", "1", "2"], [("0", "1"), ("1", "2")])
    result = compute_induced_edge_count_profile(g, 2)
    assert len(result.rows) == 2
    row0 = next(r for r in result.rows if r.edge_count == 0)
    row1 = next(r for r in result.rows if r.edge_count == 1)
    assert row0.subset_count == 1
    assert row1.subset_count == 2


def test_empty_cardinality() -> None:
    """k=0: one empty subset with 0 edges."""
    g = _graph(["a", "b"], [("a", "b")])
    result = compute_induced_edge_count_profile(g, 0)
    assert len(result.rows) == 1
    assert result.rows[0].edge_count == 0
    assert result.rows[0].subset_count == 1


def test_full_cardinality() -> None:
    """k=|V|: one subset (all vertices) with all edges."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    result = compute_induced_edge_count_profile(g, 3)
    assert len(result.rows) == 1
    assert result.rows[0].edge_count == 2
    assert result.rows[0].subset_count == 1


def test_histogram_aggregation() -> None:
    """The histogram is the aggregation of all retained rows."""
    g = _graph(["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")])
    result = compute_induced_edge_count_profile(g, 2)
    total = sum(r.subset_count for r in result.rows)
    from math import comb

    assert total == comb(4, 2)


def test_witness_replay() -> None:
    """Replay: verify the witness subset actually has the claimed edge count."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    result = compute_induced_edge_count_profile(g, 2)
    edges = set(g.edges)
    for row in result.rows:
        subset_set = set(row.witness)
        count = sum(1 for a, b in edges if a in subset_set and b in subset_set)
        assert count == row.edge_count


def test_exhaustive_comparison() -> None:
    """Compare against independent enumeration."""
    g = _graph(["a", "b", "c", "d"], [("a", "b"), ("a", "c"), ("b", "c"), ("c", "d")])
    k = 2
    result = compute_induced_edge_count_profile(g, k)
    expected: dict[int, int] = {}
    edges = set(g.edges)
    for subset in combinations(g.vertices, k):
        subset_set = set(subset)
        count = sum(1 for a, b in edges if a in subset_set and b in subset_set)
        expected[count] = expected.get(count, 0) + 1
    actual = {r.edge_count: r.subset_count for r in result.rows}
    assert actual == expected


def test_rejects_cardinality_too_large() -> None:
    g = _graph(["a", "b"], [("a", "b")])
    request = InducedEdgeCountProfileRequest(graph=g, cardinality=3)
    with pytest.raises(OperationDomainValidationError):
        compute_induced_edge_count_profile(request.graph, request.cardinality)


def test_native_admission_rejects_excessive_subset_edge_work() -> None:
    g = _graph(
        [f"{index:02}" for index in range(20)],
        [
            (f"{left:02}", f"{right:02}")
            for left in range(20)
            for right in range(left + 1, 20)
        ],
    )
    with pytest.raises(OperationDomainValidationError, match="work bound"):
        compute_induced_edge_count_profile(g, 10)


def test_rejects_excessive_retained_label_allocation() -> None:
    label = "x" * (MAX_RETAINED_LABEL_CHARACTERS // 2 + 1)
    graph = _graph((label,), ())

    with pytest.raises(OperationDomainValidationError, match="retained label"):
        compute_induced_edge_count_profile(graph, 1)


def test_result_preserves_source() -> None:
    """Result retains the source graph and cardinality."""
    g = _graph(["a", "b"], [("a", "b")])
    result = compute_induced_edge_count_profile(g, 1)
    assert result.graph == g
    assert result.cardinality == 1
