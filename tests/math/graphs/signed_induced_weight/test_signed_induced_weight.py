from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.graphs.optimization._models import (
    RationalWeightedEdge,
    RationalWeightedGraph,
)
from jacobian.math.graphs.signed_induced_weight._models import (
    SignedInducedWeightRequest,
)
from jacobian.math.graphs.signed_induced_weight.operations import (
    signed_induced_weight_extrema,
)


def _edge(a: str, b: str, w: int | str) -> RationalWeightedEdge:
    frac = Fraction(w, 1) if isinstance(w, int) else Fraction(w)
    return RationalWeightedEdge(
        endpoints=(a, b) if a < b else (b, a),
        weight=CanonicalRational.from_fraction(frac),
    )


def _graph(vertices, edges) -> RationalWeightedGraph:
    return RationalWeightedGraph(
        vertices=tuple(vertices),
        edges=tuple(edges),
    )


def _simple_graph(vertices, edge_specs) -> RationalWeightedGraph:
    return _graph(vertices, [_edge(a, b, w) for a, b, w in edge_specs])


def test_empty_graph() -> None:
    """Empty graph has min=max=0."""
    g = _simple_graph(["a"], [])
    result = signed_induced_weight_extrema(g)
    assert result.minimum.value.as_fraction() == Fraction(0)
    assert result.maximum.value.as_fraction() == Fraction(0)
    assert result.minimum.witness_vertices == ()
    assert result.maximum.witness_vertices == ()


def test_single_edge_positive() -> None:
    """Graph with one positive edge: max picks both endpoints."""
    g = _simple_graph(["a", "b"], [("a", "b", 3)])
    result = signed_induced_weight_extrema(g)
    assert result.maximum.value.as_fraction() == Fraction(3)
    assert set(result.maximum.witness_vertices) == {"a", "b"}
    assert result.minimum.value.as_fraction() == Fraction(0)


def test_single_edge_negative() -> None:
    """Graph with one negative edge: min picks both endpoints, max stays at 0."""
    g = _simple_graph(["a", "b"], [("a", "b", -5)])
    result = signed_induced_weight_extrema(g)
    assert result.minimum.value.as_fraction() == Fraction(-5)
    assert set(result.minimum.witness_vertices) == {"a", "b"}
    assert result.maximum.value.as_fraction() == Fraction(0)


def test_mixed_weights_fixture() -> None:
    """Fixture from issue: w01=2, w02=-1, w12=-1 on three vertices."""
    g = _simple_graph(
        ["0", "1", "2"],
        [("0", "1", 2), ("0", "2", -1), ("1", "2", -1)],
    )
    result = signed_induced_weight_extrema(g)
    assert result.maximum.value.as_fraction() == Fraction(2)
    assert result.minimum.value.as_fraction() == Fraction(-1)


def test_edgeless_graph() -> None:
    """Edgeless graph: min=max=0 for all subsets."""
    g = _simple_graph(["a", "b", "c"], [])
    result = signed_induced_weight_extrema(g)
    assert result.minimum.value.as_fraction() == Fraction(0)
    assert result.maximum.value.as_fraction() == Fraction(0)


def test_rational_weights() -> None:
    """Test with non-integer rational weights."""
    g = _simple_graph(["a", "b"], [("a", "b", "1/2")])
    result = signed_induced_weight_extrema(g)
    assert result.maximum.value.as_fraction() == Fraction(1, 2)


def test_witness_replay() -> None:
    """Replay the witness subset to verify the weight."""
    g = _simple_graph(
        ["a", "b", "c", "d"],
        [("a", "b", 1), ("b", "c", -2), ("c", "d", 3), ("a", "d", -1)],
    )
    result = signed_induced_weight_extrema(g)

    def replay(selected):
        total = Fraction(0)
        sset = set(selected)
        for edge in g.edges:
            a, b = edge.endpoints
            if a in sset and b in sset:
                total += edge.weight.as_fraction()
        return total

    assert replay(result.minimum.witness_vertices) == result.minimum.value.as_fraction()
    assert replay(result.maximum.witness_vertices) == result.maximum.value.as_fraction()


def test_tie_breaking_lexicographic() -> None:
    """Ties are broken by lexicographically least witness."""
    g = _simple_graph(
        ["a", "b", "c"],
        [("a", "b", 1), ("a", "c", 1), ("b", "c", 1)],
    )
    result = signed_induced_weight_extrema(g)
    assert result.maximum.value.as_fraction() == Fraction(3)
    assert set(result.maximum.witness_vertices) == {"a", "b", "c"}


def test_rejects_too_many_vertices() -> None:
    """Graph with >20 vertices should be rejected."""
    vertices = [str(i) for i in range(21)]
    g = _graph(vertices, [])
    with pytest.raises(ValidationError):
        SignedInducedWeightRequest(graph=g)
