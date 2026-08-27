"""Tests for fixed-length simple path profiles."""

import pytest

from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.math.graphs.transforms._path_profile_models import PathProfileRequest
from jacobian.math.graphs.transforms._path_profile_operations import (
    compute_path_profile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def test_path_length_0() -> None:
    graph = SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),))
    result = compute_path_profile(PathProfileRequest(graph=graph, path_length=0))
    assert len(result.rows) == 2


def test_path_length_1() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c"),
        edges=(("a", "b"), ("b", "c")),
    )
    result = compute_path_profile(PathProfileRequest(graph=graph, path_length=1))
    counts = {(r.source, r.target): r.path_count for r in result.rows}
    assert counts.get(("a", "b")) == 1
    assert counts.get(("b", "a")) == 1


def test_path_length_2() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c"),
        edges=(("a", "b"), ("b", "c")),
    )
    result = compute_path_profile(PathProfileRequest(graph=graph, path_length=2))
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

    with pytest.raises(ValueError, match="work budget"):
        PathProfileRequest(graph=graph, path_length=10)


def test_path_profile_result_budget_scales_to_requested_endpoint_pairs() -> None:
    graph = SimpleUndirectedGraph(vertices=("a" * 64, "b" * 64), edges=())

    request = PathProfileRequest(graph=graph, path_length=0)
    result = compute_path_profile(request)

    assert len(encode_strict_json(result.model_dump(mode="json"))) <= (
        CanonicalLimits().max_output_bytes
    )
    assert len(result.rows) == 2
