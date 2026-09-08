"""Tests for fixed-length simple path profiles."""

from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.transforms import path_profile
from jacobian.math.graphs.transforms._path_profile_models import PathProfileRequest
from jacobian.math.graphs.values import SimpleUndirectedGraph


def test_path_length_0() -> None:
    graph = SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),))
    result = path_profile(graph, 0)
    assert len(result.rows) == 2


def test_path_length_1() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c"),
        edges=(("a", "b"), ("b", "c")),
    )
    result = path_profile(graph, 1)
    counts = {(r.source, r.target): r.path_count for r in result.rows}
    assert counts.get(("a", "b")) == 1
    assert counts.get(("b", "a")) == 1


def test_path_length_2() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c"),
        edges=(("a", "b"), ("b", "c")),
    )
    result = path_profile(graph, 2)
    counts = {(r.source, r.target): r.path_count for r in result.rows}
    assert counts.get(("a", "c")) == 1


def test_path_profile_rejects_unbounded_dense_search() -> None:
    vertices = tuple(sorted(f"v{i}" for i in range(20)))
    edges = tuple(
        (vertices[left], vertices[right])
        for left in range(len(vertices))
        for right in range(left + 1, len(vertices))
    )
    graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)

    request = PathProfileRequest(graph=graph, path_length=10)
    with pytest.raises(OperationDomainValidationError, match="work budget"):
        path_profile(request.graph, request.path_length)


@pytest.mark.parametrize("length", [0, 2, 5])
def test_star_profile_is_admitted_by_unique_paths(length: int) -> None:
    vertices = tuple(f"{i:02d}" for i in range(64))
    graph = SimpleUndirectedGraph(
        vertices=vertices, edges=tuple((vertices[0], v) for v in vertices[1:])
    )
    result = path_profile(graph, length)
    expected_count = {0: 64, 2: 63 * 62, 5: 0}[length]
    assert len(result.rows) == expected_count
    assert all(row.path_count == 1 for row in result.rows)
    if length == 2:
        assert all(
            row.source != "00" and row.target != "00" and row.source != row.target
            for row in result.rows
        )


def test_path_profile_admits_256_vertices_and_rejects_orders_above_row_bound() -> None:
    vertices = tuple(f"{index:03d}" for index in range(256))
    edges = tuple((left, right) for left, right in combinations(vertices, 2))
    graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)
    result = path_profile(graph, 1)
    assert len(result.rows) == 256 * 255
    restored = type(result).model_validate(result.model_dump())
    assert restored == result
    PathProfileRequest(graph=graph, path_length=1)

    oversized = SimpleUndirectedGraph(
        vertices=tuple(f"{index:03d}" for index in range(257)),
        edges=tuple(
            (f"{left:03d}", f"{right:03d}")
            for left, right in combinations(range(257), 2)
        ),
    )
    with pytest.raises(ValidationError, match="at most 256 vertices"):
        PathProfileRequest(graph=oversized, path_length=1)
    with pytest.raises(OperationDomainValidationError, match="at most 256 vertices"):
        path_profile(oversized, 1)
