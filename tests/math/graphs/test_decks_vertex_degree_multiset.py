"""Exact degree-multiset reconstruction from complete vertex decks."""

from itertools import combinations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.decks import (
    UnlabelledVertexDeck,
    VertexDeletionFamily,
    unlabelled_vertex_deck,
    vertex_deck_degree_multiset,
    vertex_deletion_family,
)
from jacobian.math.graphs.realization._models import DegreeSequence
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _all_graphs(order: int):
    vertices = tuple(f"v{index}" for index in range(order))
    possible_edges = tuple(combinations(vertices, 2))
    for mask in range(1 << len(possible_edges)):
        edges = tuple(
            edge for bit, edge in enumerate(possible_edges) if mask & (1 << bit)
        )
        yield SimpleUndirectedGraph(vertices=vertices, edges=edges)


def _direct_degree_multiset(graph: SimpleUndirectedGraph) -> tuple[int, ...]:
    """Independent adjacency count, without using any deck/card operation."""
    return tuple(
        sorted(
            (sum(vertex in edge for edge in graph.edges) for vertex in graph.vertices),
            reverse=True,
        )
    )


@pytest.mark.parametrize("order", [3, 4, 5])
def test_every_labeled_graph_through_order_five_matches_direct_oracle(
    order: int,
) -> None:
    for graph in _all_graphs(order):
        expected = _direct_degree_multiset(graph)
        deck = unlabelled_vertex_deck(vertex_deletion_family(graph))

        result = vertex_deck_degree_multiset(deck)

        assert type(result) is DegreeSequence
        assert result.degrees == expected
        assert type(result).model_validate_json(result.model_dump_json()) == result


def test_repeated_cards_and_independent_relabelling_preserve_degree_multiset() -> None:
    triangle = SimpleUndirectedGraph(
        vertices=("a", "b", "c"),
        edges=(("a", "b"), ("a", "c"), ("b", "c")),
    )
    deck = unlabelled_vertex_deck(vertex_deletion_family(triangle))
    assert tuple(card_class.multiplicity for card_class in deck.classes) == (3,)
    assert vertex_deck_degree_multiset(deck).degrees == (2, 2, 2)

    relabelled = SimpleUndirectedGraph(
        vertices=("x", "y", "z"),
        edges=(("x", "y"), ("x", "z"), ("y", "z")),
    )
    relabelled_deck = unlabelled_vertex_deck(vertex_deletion_family(relabelled))
    assert vertex_deck_degree_multiset(relabelled_deck).degrees == (2, 2, 2)


def test_rejects_a_forged_card_multiplicity_before_using_it() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c"))
    )
    deck = unlabelled_vertex_deck(vertex_deletion_family(graph))
    first_class = deck.classes[0].model_copy(
        update={"multiplicity": deck.classes[0].multiplicity + 1}
    )
    forged = UnlabelledVertexDeck.model_construct(
        family=deck.family,
        classes=(first_class, *deck.classes[1:]),
        card_count=deck.card_count,
    )

    with pytest.raises(OperationDomainValidationError, match="exact multiset quotient"):
        vertex_deck_degree_multiset(forged)


def test_small_order_boundary_and_empty_graph_on_three_vertices() -> None:
    small = SimpleUndirectedGraph(vertices=("a", "b"), edges=())
    small_deck = unlabelled_vertex_deck(vertex_deletion_family(small))
    with pytest.raises(OperationDomainValidationError, match="at least three"):
        vertex_deck_degree_multiset(small_deck)

    empty = SimpleUndirectedGraph(vertices=("a", "b", "c"), edges=())
    empty_deck = unlabelled_vertex_deck(vertex_deletion_family(empty))
    assert vertex_deck_degree_multiset(empty_deck).degrees == (0, 0, 0)


def test_admits_degree_output_bound_before_deck_work() -> None:
    source = SimpleUndirectedGraph(
        vertices=tuple(f"v{index}" for index in range(11)), edges=()
    )
    forged_family = VertexDeletionFamily.model_construct(
        source=source, cards=(), edge_appearances=(), vertex_appearances=()
    )
    forged_deck = UnlabelledVertexDeck.model_construct(
        family=forged_family, classes=(), card_count=0
    )
    with pytest.raises(OperationResourceAdmissionError, match="output exceeds"):
        vertex_deck_degree_multiset(forged_deck)
