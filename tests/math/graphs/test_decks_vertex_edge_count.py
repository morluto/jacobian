"""Exact vertex-deck edge-count reconstruction and independent small oracle."""

from itertools import combinations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.decks.operations import (
    unlabelled_vertex_deck,
    vertex_deck_edge_count,
    vertex_deletion_family,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _independent_card_edge_counts(graph: SimpleUndirectedGraph) -> tuple[int, ...]:
    """Count retained edges directly from the source edge list per deleted vertex."""
    return tuple(
        sorted(
            sum(vertex not in edge for edge in graph.edges) for vertex in graph.vertices
        )
    )


def _all_graphs(order: int):
    vertices = tuple(f"v{index}" for index in range(order))
    possible_edges = tuple(combinations(vertices, 2))
    for mask in range(1 << len(possible_edges)):
        edges = tuple(
            edge for bit, edge in enumerate(possible_edges) if mask & (1 << bit)
        )
        yield SimpleUndirectedGraph(vertices=vertices, edges=edges)


@pytest.mark.parametrize("order", [3, 4])
def test_all_graphs_through_order_four_against_direct_deletion_oracle(
    order: int,
) -> None:
    for graph in _all_graphs(order):
        direct_counts = _independent_card_edge_counts(graph)
        assert sum(direct_counts) == (order - 2) * len(graph.edges)

        family = vertex_deletion_family(graph)
        deck = unlabelled_vertex_deck(family)
        result = vertex_deck_edge_count(deck)

        assert result.card_edge_counts == direct_counts
        assert result.card_edge_total == sum(direct_counts)
        assert result.overcount_divisor == order - 2
        assert result.source_edge_count == len(graph.edges)


def test_triangle_card_multiset_retains_repeated_class_multiplicity() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c"),
        edges=(("a", "b"), ("a", "c"), ("b", "c")),
    )
    deck = unlabelled_vertex_deck(vertex_deletion_family(graph))
    result = vertex_deck_edge_count(deck)
    assert tuple(card_class.multiplicity for card_class in deck.classes) == (3,)
    assert result.card_edge_counts == (1, 1, 1)
    assert result.source_edge_count == 3
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_rejects_noncanonical_claimed_card_classes() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c"))
    )
    deck = unlabelled_vertex_deck(vertex_deletion_family(graph))
    forged = deck.model_construct(
        family=deck.family,
        classes=deck.classes[:1],
        card_count=deck.card_count,
    )
    with pytest.raises(OperationDomainValidationError, match="exact multiset quotient"):
        vertex_deck_edge_count(forged)


def test_rejects_source_order_below_kelly_edge_count_boundary() -> None:
    graph = SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),))
    deck = unlabelled_vertex_deck(vertex_deletion_family(graph))
    with pytest.raises(OperationDomainValidationError, match="at least three"):
        vertex_deck_edge_count(deck)
