from __future__ import annotations

from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.decks._models import (
    AnonymousGraphCardMultiset,
    AnonymousGraphCardMultisetRequest,
)
from jacobian.math.graphs.decks.anonymous_vertex_degree_multiset import (
    AnonymousVertexDeckDegreeMultiset,
    anonymous_vertex_deck_degree_multiset,
)
from jacobian.math.graphs.decks.anonymous_vertex_degree_multiset._tools import TOOLS
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


def _source_degrees(graph: SimpleUndirectedGraph) -> tuple[int, ...]:
    degrees = dict.fromkeys(graph.vertices, 0)
    for left, right in graph.edges:
        degrees[left] += 1
        degrees[right] += 1
    return tuple(sorted(degrees.values(), reverse=True))


def test_exhaustive_small_graphs_recover_direct_source_degree_multiset() -> None:
    empty_graph = SimpleUndirectedGraph(vertices=(), edges=())
    assert (
        anonymous_vertex_deck_degree_multiset(_vertex_deck(empty_graph)).degrees == ()
    )
    for order in range(1, 5):
        for edge_mask in range(1 << (order * (order - 1) // 2)):
            source = _graph(order, edge_mask)
            if order == 2:
                with pytest.raises(OperationDomainValidationError):
                    anonymous_vertex_deck_degree_multiset(_vertex_deck(source))
                continue
            result = anonymous_vertex_deck_degree_multiset(_vertex_deck(source))
            assert result.degrees == _source_degrees(source)
            assert sum(result.degrees) == 2 * result.edge_count.source_edge_count


def test_divisible_but_nongraphical_degree_profile_is_rejected() -> None:
    cards = (_graph(3, 0), _graph(3, 0), _graph(3, 0b011), _graph(3, 0b011))
    deck = anonymous_graph_card_multiset(
        AnonymousGraphCardMultisetRequest(card_order=3, cards=cards)
    )
    with pytest.raises(OperationDomainValidationError, match="nongraphical"):
        anonymous_vertex_deck_degree_multiset(deck)


def test_result_round_trip_preserves_composed_edge_count_and_degrees() -> None:
    result = anonymous_vertex_deck_degree_multiset(_vertex_deck(_graph(4, 0b101101)))
    restored = AnonymousVertexDeckDegreeMultiset.model_validate_json(
        result.model_dump_json()
    )
    assert restored == result
    assert restored.edge_count.deck == _vertex_deck(_graph(4, 0b101101))


def test_operation_manifest_uses_the_existing_anonymous_deck_carrier() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "graph.deck.anonymous_vertex.degree_multiset.compute"
    )
    assert tool.request_type is AnonymousGraphCardMultiset
    result = tool.run(_vertex_deck(_graph(3, 0b011)))
    assert result.degrees == (2, 1, 1)


def test_decoded_degree_values_remain_bounded() -> None:
    result = anonymous_vertex_deck_degree_multiset(_vertex_deck(_graph(3, 0b011)))
    for degrees in ((99, 1, 1), (1, 2), (1, 2, 0), ()):
        with pytest.raises(ValidationError):
            AnonymousVertexDeckDegreeMultiset.model_validate(
                {
                    "edge_count": result.edge_count.model_dump(mode="python"),
                    "degrees": degrees,
                }
            )
