from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.graphs.optimization.signed_induced_weight.operations import (
    compute_signed_induced_weight_extrema,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _cr(num, den=1):
    return CanonicalRational.from_fraction(Fraction(num, den))


def _graph(vertices, edges):
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple(tuple(e) for e in edges),
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
