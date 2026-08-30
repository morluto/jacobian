from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.common_neighbor_profile._models import (
    CommonNeighborProfileRequest,
)
from jacobian.math.graphs.common_neighbor_profile.operations import (
    compute_common_neighbor_profile,
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
    result = compute_common_neighbor_profile(graph)
    assert result.max_codegree == 1
    assert result.is_c4_free


def test_edgeless_graph() -> None:
    graph = _graph(["a", "b", "c"], [])
    result = compute_common_neighbor_profile(graph)
    assert result.max_codegree == 0
    assert result.is_c4_free


def test_k4() -> None:
    graph = _graph(
        ["a", "b", "c", "d"],
        [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["b", "d"], ["c", "d"]],
    )
    result = compute_common_neighbor_profile(graph)
    assert result.max_codegree == 2  # Any pair has 2 common neighbors in K4


def test_c4_not_c4_free() -> None:
    graph = _graph(
        ["a", "b", "c", "d"],
        [["a", "b"], ["b", "c"], ["c", "d"], ["a", "d"]],
    )
    result = compute_common_neighbor_profile(graph)
    # a-c have common neighbors b,d -> codegree 2
    assert not result.is_c4_free
    assert result.max_codegree == 2


def test_path_graph() -> None:
    graph = _graph(["a", "b", "c", "d"], [["a", "b"], ["b", "c"], ["c", "d"]])
    result = compute_common_neighbor_profile(graph)
    assert result.is_c4_free


def test_result_preserves_source() -> None:
    graph = _graph(["a", "b"], [["a", "b"]])
    result = compute_common_neighbor_profile(graph)
    assert result.graph == graph


def test_dense_profile_is_rejected_from_exact_incidence_bound() -> None:
    vertices = [f"v{i:03d}" for i in range(256)]
    edges = [
        [left, right]
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
    ]
    graph = _graph(vertices, edges)

    with pytest.raises(OperationDomainValidationError, match="output budget"):
        compute_common_neighbor_profile(graph)
    with pytest.raises(ValidationError, match="output budget"):
        CommonNeighborProfileRequest(graph=graph)
