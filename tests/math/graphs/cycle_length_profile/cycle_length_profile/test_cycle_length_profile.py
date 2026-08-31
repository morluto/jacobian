from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.cycle_length_profile._models import (
    CycleLengthProfileRequest,
)
from jacobian.math.graphs.cycle_length_profile.operations import (
    compute_cycle_length_profile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(
    vertices: Sequence[str], edges: Sequence[Sequence[str]]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((edge[0], edge[1]) for edge in edges),
    )


def test_triangle() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
    result = compute_cycle_length_profile(graph)
    assert tuple(r.cycle_length for r in result.rows) == (3,)


def test_edgeless_graph() -> None:
    graph = _graph(["a", "b", "c"], [])
    result = compute_cycle_length_profile(graph)
    assert tuple(r.cycle_length for r in result.rows) == ()


def test_path_graph() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"]])
    result = compute_cycle_length_profile(graph)
    assert tuple(r.cycle_length for r in result.rows) == ()


def test_k4() -> None:
    graph = _graph(
        ["a", "b", "c", "d"],
        [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["b", "d"], ["c", "d"]],
    )
    result = compute_cycle_length_profile(graph)
    assert set(tuple(r.cycle_length for r in result.rows)) == {3, 4}


def test_result_preserves_source() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
    result = compute_cycle_length_profile(graph)
    assert result.graph == graph


def test_dense_bipartite_search_is_rejected_before_traversal() -> None:
    left = [f"l{i}" for i in range(20)]
    right = [f"r{i}" for i in range(20)]
    graph = _graph([*left, *right], [[u, v] for u in left for v in right])

    with pytest.raises(OperationDomainValidationError, match="cycle-profile search exceeds the admitted work bound"):
        compute_cycle_length_profile(graph)



def test_large_single_cycle_remains_admitted() -> None:
    vertices = [f"v{i:02d}" for i in range(20)]
    edges = [[vertices[i], vertices[i + 1]] for i in range(19)]
    edges.append([vertices[0], vertices[-1]])

    result = compute_cycle_length_profile(_graph(vertices, edges))

    assert tuple(r.cycle_length for r in result.rows) == (20,)


def test_wide_cycle_labels_are_rejected_before_search() -> None:
    labels = tuple(f"{prefix}{'x' * 1_000_000}" for prefix in ("a", "b", "c"))
    graph = _graph(
        labels, [[labels[0], labels[1]], [labels[1], labels[2]], [labels[0], labels[2]]]
    )

    with pytest.raises(OperationDomainValidationError, match="exceeds the canonical output bound"):
        compute_cycle_length_profile(graph)
