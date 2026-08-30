from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.optimization.signed_induced_weight.operations import (
    compute_signed_induced_weight_extrema,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _cr(num: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(num, den))


def _graph(
    vertices: Sequence[str], edges: Sequence[Sequence[str]]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((edge[0], edge[1]) for edge in edges),
    )


def test_empty_subset() -> None:
    graph = _graph(["a", "b"], [["a", "b"]])
    result = compute_signed_induced_weight_extrema(graph, (("a", "b", _cr(1)),))
    # Empty subset has weight 0
    assert result.minimum_weight.as_fraction() == Fraction(0)


def test_positive_weight() -> None:
    graph = _graph(["a", "b"], [["a", "b"]])
    result = compute_signed_induced_weight_extrema(graph, (("a", "b", _cr(5)),))
    # Best: include both a and b -> weight 5
    assert result.maximum_weight.as_fraction() == Fraction(5)
    assert result.maximum_witness == ("a", "b")


def test_negative_weight() -> None:
    graph = _graph(["a", "b"], [["a", "b"]])
    result = compute_signed_induced_weight_extrema(graph, (("a", "b", _cr(-5)),))
    # Best: include both -> weight -5
    assert result.minimum_weight.as_fraction() == Fraction(-5)
    assert result.minimum_witness == ("a", "b")


def test_mixed_weights() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"]])
    result = compute_signed_induced_weight_extrema(
        graph, (("a", "b", _cr(1)), ("b", "c", _cr(-1)))
    )
    # Subsets: {}, {a}, {b}, {c}, {a,b}=1, {a,c}=0, {b,c}=-1, {a,b,c}=0
    assert result.maximum_weight.as_fraction() == Fraction(1)
    assert result.minimum_weight.as_fraction() == Fraction(-1)


def test_result_preserves_source() -> None:
    graph = _graph(["a", "b"], [["a", "b"]])
    result = compute_signed_induced_weight_extrema(graph, (("a", "b", _cr(1)),))
    assert result.graph == graph
    assert result.edge_weights == (("a", "b", _cr(1)),)


def test_nonedge_weight_is_rejected() -> None:
    graph = _graph(["a", "b"], [])

    with pytest.raises(OperationDomainValidationError, match="one-for-one"):
        compute_signed_induced_weight_extrema(graph, (("a", "b", _cr(10)),))


def test_uniform_sign_large_graph_uses_direct_extrema() -> None:
    vertices = tuple(f"{index:02d}" for index in range(64))
    edges = tuple((vertices[index], vertices[index + 1]) for index in range(63))
    graph = _graph(vertices, edges)
    weights = tuple((left, right, _cr(1)) for left, right in graph.edges)

    result = compute_signed_induced_weight_extrema(graph, weights)

    assert result.maximum_weight.as_fraction() == 63
