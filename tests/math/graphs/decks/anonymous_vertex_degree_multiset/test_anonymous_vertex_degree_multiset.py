from __future__ import annotations

from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.decks._models import (
    AnonymousGraphCardMultiset,
)
from jacobian.math.graphs.decks.anonymous_vertex_degree_multiset import (
    anonymous_vertex_deck_degree_multiset,
)
from jacobian.math.graphs.decks.operations import anonymous_graph_card_multiset
from jacobian.math.graphs.realization._models import DegreeSequence
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
        card_order=max(0, len(graph.vertices) - 1), cards=tuple(cards)
    )


def _source_degrees(graph: SimpleUndirectedGraph) -> tuple[int, ...]:
    degrees = dict.fromkeys(graph.vertices, 0)
    for left, right in graph.edges:
        degrees[left] += 1
        degrees[right] += 1
    return tuple(sorted(degrees.values(), reverse=True))


def test_exhaustive_small_graphs_recover_canonical_degree_sequences() -> None:
    empty_graph = SimpleUndirectedGraph(vertices=(), edges=())
    assert anonymous_vertex_deck_degree_multiset(_vertex_deck(empty_graph)) == DegreeSequence(
        degrees=()
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
            assert sum(result.degrees) == 2 * len(source.edges)


def test_divisible_but_nongraphical_degree_profile_is_rejected() -> None:
    cards = (_graph(3, 0), _graph(3, 0), _graph(3, 0b011), _graph(3, 0b011))
    deck = anonymous_graph_card_multiset(card_order=3, cards=cards)
    with pytest.raises(OperationDomainValidationError, match="nongraphical"):
        anonymous_vertex_deck_degree_multiset(deck)


def test_result_is_the_canonical_reusable_degree_sequence() -> None:
    deck = _vertex_deck(_graph(4, 0b101101))
    result = anonymous_vertex_deck_degree_multiset(deck)
    assert type(result) is DegreeSequence
    assert DegreeSequence.model_validate_json(result.model_dump_json()) == result


def test_canonical_degree_sequence_retains_global_scalar_bounds() -> None:
    for degrees in ((99, 1, 1),):
        with pytest.raises(ValidationError):
            DegreeSequence.model_validate({"degrees": degrees})
