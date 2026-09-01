from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.rainbow_embedding.operations import (
    compute_rainbow_embedding_profile,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph, SimpleUndirectedGraph


def _p2():
    return SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),))


def _k3_all_distinct():
    return ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=("0", "1", "2"),
            edges=(("0", "1"), ("0", "2"), ("1", "2")),
        ),
        edge_colors=("red", "blue", "green"),
    )


def _k3_two_same():
    return ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=("0", "1", "2"),
            edges=(("0", "1"), ("0", "2"), ("1", "2")),
        ),
        edge_colors=("red", "red", "blue"),
    )


def test_p2_all_distinct_rainbow() -> None:
    """P2 in K3 with all-distinct colours: all 6 embeddings are rainbow."""
    result = compute_rainbow_embedding_profile(_p2(), _k3_all_distinct())
    assert result.total_embeddings == 6
    assert result.rainbow_count == 6


def test_p2_two_same() -> None:
    """P2 in K3 with two same colours: 6 total, all rainbow (single edge) (the ones using edge 1-2 which is blue)."""
    result = compute_rainbow_embedding_profile(_p2(), _k3_two_same())
    assert result.total_embeddings == 6
    assert result.rainbow_count == 6  # single edge is always rainbow


def test_empty_pattern() -> None:
    """The empty pattern has one empty injective embedding."""
    empty = SimpleUndirectedGraph(vertices=(), edges=())
    result = compute_rainbow_embedding_profile(empty, _k3_all_distinct())
    assert result.total_embeddings == 1
    assert result.rainbow_count == 1
    assert result.embeddings[0].pattern_to_host == ()


def test_rejects_uncolored_nonempty_host() -> None:
    host = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=("0", "1"), edges=(("0", "1"),))
    )
    with pytest.raises(OperationDomainValidationError):
        compute_rainbow_embedding_profile(_p2(), host)


def test_rejects_unencodable_pattern_label() -> None:
    pattern = SimpleUndirectedGraph(vertices=("\ud800",), edges=())

    with pytest.raises(OperationDomainValidationError, match="Unicode scalar"):
        compute_rainbow_embedding_profile(pattern, _k3_all_distinct())


def test_rejects_unbounded_embedding_family() -> None:
    pattern = SimpleUndirectedGraph(vertices=tuple(str(i) for i in range(8)), edges=())
    host = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=tuple(str(i) for i in range(16)), edges=())
    )
    with pytest.raises(OperationDomainValidationError):
        compute_rainbow_embedding_profile(pattern, host)


def test_degree_impossible_pattern_returns_empty_profile() -> None:
    pattern = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "center", "d"),
        edges=(
            ("a", "center"),
            ("b", "center"),
            ("c", "center"),
            ("center", "d"),
        ),
    )
    host = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=tuple(f"v{index:02d}" for index in range(13)),
            edges=tuple((f"v{index:02d}", f"v{index + 1:02d}") for index in range(12)),
        ),
        edge_colors=tuple("red" for _ in range(12)),
    )

    result = compute_rainbow_embedding_profile(pattern, host)

    assert result.total_embeddings == 0
    assert result.embeddings == ()


def test_budget_admission_allows_cheap_seventeen_vertex_host() -> None:
    pattern = SimpleUndirectedGraph(vertices=("v",), edges=())
    host = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=tuple(f"v{index}" for index in range(17)), edges=()
        )
    )

    result = compute_rainbow_embedding_profile(pattern, host)

    assert result.total_embeddings == 17


def test_pattern_larger_than_host() -> None:
    """Pattern larger than host: 0 embeddings."""
    large = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(("a", "b"), ("b", "c"), ("c", "d")),
    )
    result = compute_rainbow_embedding_profile(large, _k3_all_distinct())
    assert result.total_embeddings == 0


def test_edgeless_pattern() -> None:
    """Edgeless pattern: all injective maps, all vacuously rainbow."""
    single = SimpleUndirectedGraph(vertices=("a",), edges=())
    result = compute_rainbow_embedding_profile(single, _k3_all_distinct())
    assert result.total_embeddings == 3
    assert result.rainbow_count == 3


def test_result_preserves_source() -> None:
    result = compute_rainbow_embedding_profile(_p2(), _k3_all_distinct())
    assert result.pattern == _p2()
    assert result.host == _k3_all_distinct()
