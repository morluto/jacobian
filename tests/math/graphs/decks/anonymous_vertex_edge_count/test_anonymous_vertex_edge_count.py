from __future__ import annotations

from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.decks._models import (
    AnonymousGraphCardMultiset,
    AnonymousGraphCardMultisetRequest,
)
from jacobian.math.graphs.decks.anonymous_vertex_edge_count import (
    AnonymousVertexDeckEdgeCount,
    anonymous_vertex_deck_edge_count,
)
from jacobian.math.graphs.decks.anonymous_vertex_edge_count._tools import TOOLS
from jacobian.math.graphs.decks.operations import anonymous_graph_card_multiset
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(order: int, edge_mask: int) -> SimpleUndirectedGraph:
    vertices = tuple(f"x{i}" for i in range(order))
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


def test_exhaustive_small_graphs_recover_direct_source_edge_count() -> None:
    empty_deck = anonymous_graph_card_multiset(
        AnonymousGraphCardMultisetRequest(card_order=0, cards=())
    )
    assert anonymous_vertex_deck_edge_count(empty_deck).source_edge_count == 0
    for order in range(1, 5):
        for mask in range(1 << (order * (order - 1) // 2)):
            source = _graph(order, mask)
            if order == 2:
                with pytest.raises(OperationDomainValidationError):
                    anonymous_vertex_deck_edge_count(_vertex_deck(source))
                continue
            result = anonymous_vertex_deck_edge_count(_vertex_deck(source))
            assert result.source_order == order
            assert result.source_edge_count == len(source.edges)
            assert result.card_edge_total == sum(
                len(item.representative.edges) * item.multiplicity
                for item in result.deck.classes
            )


def test_order_two_vertex_deck_does_not_determine_source_edge_count() -> None:
    empty = _vertex_deck(_graph(2, 0))
    one_edge = _vertex_deck(_graph(2, 1))
    assert empty == one_edge
    with pytest.raises(OperationDomainValidationError, match="do not determine"):
        anonymous_vertex_deck_edge_count(empty)


def test_nondivisible_card_edge_total_is_rejected() -> None:
    cards = (_graph(3, 1), _graph(3, 0), _graph(3, 0), _graph(3, 0))
    deck = anonymous_graph_card_multiset(
        AnonymousGraphCardMultisetRequest(card_order=3, cards=cards)
    )
    with pytest.raises(OperationDomainValidationError, match="not divisible"):
        anonymous_vertex_deck_edge_count(deck)


def test_result_round_trip_preserves_typed_input_deck_and_quotient() -> None:
    result = anonymous_vertex_deck_edge_count(_vertex_deck(_graph(4, 0b101101)))
    restored = AnonymousVertexDeckEdgeCount.model_validate_json(
        result.model_dump_json()
    )
    assert restored == result
    with pytest.raises(ValidationError, match="Kelly edge quotient"):
        AnonymousVertexDeckEdgeCount.model_validate(
            {
                **result.model_dump(mode="python"),
                "source_edge_count": result.source_edge_count + 1,
            }
        )


def test_catalog_publishes_operation_composable_from_anonymous_card_carrier() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "graph.deck.anonymous_vertex.edge_count.compute"
    )
    assert tool.request_type.__name__ == "AnonymousGraphCardMultiset"
    result = tool.run(_vertex_deck(_graph(3, 0b011)))
    assert result.source_edge_count == 2
