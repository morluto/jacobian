from __future__ import annotations

from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.decks._models import (
    AnonymousGraphCardClass,
    AnonymousGraphCardMultiset,
    AnonymousGraphCardMultisetRequest,
)
from jacobian.math.graphs.decks.anonymous_vertex_edge_count.operations import (
    anonymous_vertex_deck_edge_count,
)
from jacobian.math.graphs.decks.anonymous_vertex_source_order import (
    AnonymousVertexDeckSourceOrder,
    anonymous_vertex_deck_source_order,
)
from jacobian.math.graphs.decks.anonymous_vertex_source_order._tools import TOOLS
from jacobian.math.graphs.decks.operations import anonymous_graph_card_multiset
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(order: int, edge_mask: int) -> SimpleUndirectedGraph:
    vertices = tuple(f"x{index}" for index in range(order))
    pairs = tuple(combinations(vertices, 2))
    edges = tuple(pair for bit, pair in enumerate(pairs) if edge_mask & (1 << bit))
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def _vertex_deck(graph: SimpleUndirectedGraph) -> AnonymousGraphCardMultiset:
    cards = []
    for deleted in graph.vertices:
        retained = tuple(vertex for vertex in graph.vertices if vertex != deleted)
        edges = tuple(edge for edge in graph.edges if deleted not in edge)
        cards.append(SimpleUndirectedGraph(vertices=retained, edges=edges))
    return anonymous_graph_card_multiset(
        AnonymousGraphCardMultisetRequest(
            card_order=max(0, len(graph.vertices) - 1), cards=tuple(cards)
        )
    )


def test_exhaustive_small_simple_graph_decks_recover_source_order() -> None:
    empty = SimpleUndirectedGraph(vertices=(), edges=())
    assert anonymous_vertex_deck_source_order(_vertex_deck(empty)).source_order == 0
    for order in range(1, 5):
        for edge_mask in range(1 << (order * (order - 1) // 2)):
            deck = _vertex_deck(_graph(order, edge_mask))
            assert anonymous_vertex_deck_source_order(deck).source_order == order


def test_order_two_is_recovered_even_when_edge_count_is_not() -> None:
    deck = _vertex_deck(_graph(2, 1))
    result = anonymous_vertex_deck_source_order(deck)
    assert result.source_order == 2
    assert sum(card_class.multiplicity for card_class in result.deck.classes) == 2


@pytest.mark.parametrize(
    ("card_order", "multiplicity"),
    ((0, 2), (2, 2), (2, 4), (4, 1)),
)
def test_incomplete_or_inconsistent_card_multisets_are_rejected(
    card_order: int, multiplicity: int
) -> None:
    representative = SimpleUndirectedGraph(
        vertices=tuple(f"v{index:02}" for index in range(card_order)), edges=()
    )
    deck = AnonymousGraphCardMultiset.model_construct(
        card_order=card_order,
        classes=(
            AnonymousGraphCardClass(
                representative=representative,
                multiplicity=multiplicity,
            ),
        ),
    )
    with pytest.raises((OperationDomainValidationError, ValidationError)):
        anonymous_vertex_deck_source_order(deck)


def test_result_roundtrip_retains_the_anonymous_deck_context() -> None:
    deck = _vertex_deck(_graph(4, 0b101101))
    result = anonymous_vertex_deck_source_order(deck)
    restored = AnonymousVertexDeckSourceOrder.model_validate_json(
        result.model_dump_json()
    )
    assert restored == result
    assert restored.deck == deck
    assert anonymous_vertex_deck_edge_count(restored.deck).source_order == 4


def test_operation_declaration_composes_with_the_existing_card_carrier() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "graph.deck.anonymous_vertex.source_order.compute"
    )
    assert tool.request_type is AnonymousGraphCardMultiset
    assert tool.result_type is AnonymousVertexDeckSourceOrder
    assert tool.run(_vertex_deck(_graph(3, 0b011))).source_order == 3
