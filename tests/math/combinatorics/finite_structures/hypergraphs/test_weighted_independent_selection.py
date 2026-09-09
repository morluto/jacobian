"""Rational-weighted hypergraph independent selection tests."""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._weighted_independence import (
    WeightedIndependentSelectionRequest,
    maximum_weight_independent_selection,
)


def _rational(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def test_exact_weighted_selection_matches_small_oracle() -> None:
    request = WeightedIndependentSelectionRequest(
        hypergraph=FiniteHypergraph(
            vertices=("a", "b", "c"),
            edges=(("ab", ("a", "b")), ("bc", ("b", "c"))),
        ),
        weights=(_rational(2), _rational(3), _rational(2)),
    )
    result = maximum_weight_independent_selection(request)
    assert result.status == "EXACT"
    assert result.incumbent_vertices == ("a", "c")
    assert result.incumbent_value.as_fraction() == 4
    assert result.lower_bound == result.upper_bound


def test_node_limit_returns_attaining_source_bound_result() -> None:
    request = WeightedIndependentSelectionRequest(
        hypergraph=FiniteHypergraph(vertices=("a", "b"), edges=()),
        weights=(_rational(1), _rational(2)),
        search_node_limit=2,
    )
    result = maximum_weight_independent_selection(request)
    assert result.status == "BOUNDED"
    assert result.lower_bound.as_fraction() <= result.upper_bound.as_fraction()
    assert result.incumbent_value == result.lower_bound


def test_negative_weights_preserve_empty_incumbent() -> None:
    request = WeightedIndependentSelectionRequest(
        hypergraph=FiniteHypergraph(vertices=("a",), edges=()),
        weights=(_rational(-1),),
    )
    result = maximum_weight_independent_selection(request)
    assert result.incumbent_vertices == ()
    assert result.incumbent_value.as_fraction() == 0
