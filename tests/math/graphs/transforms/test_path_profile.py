"""Tests for fixed-length simple path profiles."""

from jacobian.math.graphs.transforms._path_profile_models import PathProfileRequest
from jacobian.math.graphs.transforms._path_profile_operations import compute_path_profile
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
