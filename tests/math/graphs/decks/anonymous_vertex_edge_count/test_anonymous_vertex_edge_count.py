from __future__ import annotations

from itertools import combinations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.decks._models import (
    AnonymousGraphCardClass,
    AnonymousGraphCardMultiset,
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
        card_order=max(0, len(graph.vertices) - 1), cards=tuple(cards)
    )


def test_exhaustive_small_graphs_recover_direct_realizing_edge_count() -> None:
    empty_deck = anonymous_graph_card_multiset(card_order=0, cards=())
    assert anonymous_vertex_deck_edge_count(empty_deck).realizing_edge_count == 0
    for order in range(1, 5):
        for mask in range(1 << (order * (order - 1) // 2)):
            source = _graph(order, mask)
            if order == 2:
                with pytest.raises(OperationDomainValidationError):
                    anonymous_vertex_deck_edge_count(_vertex_deck(source))
                continue
            result = anonymous_vertex_deck_edge_count(_vertex_deck(source))
            assert result.source_order == order
            assert result.realizing_edge_count == len(source.edges)
            assert result.card_edge_total == sum(
                len(item.representative.edges) * item.multiplicity
                for item in result.deck.classes
            )


def test_order_two_vertex_deck_does_not_determine_realizing_edge_count() -> None:
    empty = _vertex_deck(_graph(2, 0))
    one_edge = _vertex_deck(_graph(2, 1))
    assert empty == one_edge
    with pytest.raises(OperationDomainValidationError, match="do not determine"):
        anonymous_vertex_deck_edge_count(empty)


def test_nondivisible_card_edge_total_is_rejected() -> None:
    cards = (_graph(3, 1), _graph(3, 0), _graph(3, 0), _graph(3, 0))
    deck = anonymous_graph_card_multiset(card_order=3, cards=cards)
    with pytest.raises(OperationDomainValidationError, match="not divisible"):
        anonymous_vertex_deck_edge_count(deck)


def test_result_round_trip_preserves_typed_input_deck_and_quotient() -> None:
    result = anonymous_vertex_deck_edge_count(_vertex_deck(_graph(4, 0b101101)))
    restored = AnonymousVertexDeckEdgeCount.model_validate_json(
        result.model_dump_json()
    )
    assert restored == result
    # Transport decoding checks canonical structure, not the producer's
    # mathematical quotient; the operation itself owns that computation.
    forged = AnonymousVertexDeckEdgeCount.model_validate(
        {
            **result.model_dump(mode="python"),
            "realizing_edge_count": result.realizing_edge_count + 1,
        }
    )
    assert forged.realizing_edge_count == result.realizing_edge_count + 1
    # An authored transport result is not an authenticated computation. The
    # operation re-establishes the quotient from its retained canonical deck.
    assert anonymous_vertex_deck_edge_count(forged.deck) == result


def test_catalog_publishes_operation_composable_from_anonymous_card_carrier() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "graph.deck.anonymous_vertex.edge_count.compute"
    )
    assert tool.request_type.__name__ == "AnonymousGraphCardMultiset"
    result = tool.run(_vertex_deck(_graph(3, 0b011)))
    assert result.realizing_edge_count == 2


def test_order_nine_edgeless_deck_is_accepted_without_canonicalization() -> None:
    deck = AnonymousGraphCardMultiset.model_validate(
        {
            "card_order": 8,
            "classes": (
                {
                    "representative": {
                        "vertices": [f"v{i:02d}" for i in range(8)],
                        "edges": [],
                    },
                    "multiplicity": 9,
                },
            ),
        }
    )
    result = anonymous_vertex_deck_edge_count(deck)
    assert result.source_order == 9
    assert result.realizing_edge_count == 0


def test_full_carrier_order_ten_edgeless_deck_is_admitted() -> None:
    graph = SimpleUndirectedGraph(
        vertices=tuple(f"v{index:02d}" for index in range(10)), edges=()
    )
    deck = AnonymousGraphCardMultiset(
        card_order=10,
        classes=(AnonymousGraphCardClass(representative=graph, multiplicity=11),),
    )

    result = anonymous_vertex_deck_edge_count(deck)

    assert result.source_order == 11
    assert result.card_edge_total == 0
    assert result.realizing_edge_count == 0


@pytest.mark.parametrize("missing", ["vertices", "edges"])
def test_native_admission_rejects_missing_constructed_graph_fields(
    missing: str,
) -> None:
    fields: dict[str, object] = {"vertices": ("v00", "v01"), "edges": ()}
    fields.pop(missing)
    graph = SimpleUndirectedGraph.model_construct(_fields_set=None, **fields)
    item = AnonymousGraphCardClass.model_construct(representative=graph, multiplicity=3)
    deck = AnonymousGraphCardMultiset.model_construct(card_order=2, classes=(item,))

    with pytest.raises(
        OperationDomainValidationError, match="bounded graph representatives"
    ):
        anonymous_vertex_deck_edge_count(deck)


def test_native_admission_rejects_oversized_edge_tuple_before_edge_scan() -> None:
    graph = SimpleUndirectedGraph.model_construct(
        vertices=("v00", "v01"), edges=(("v00", "v01"),) * 100_000
    )
    item = AnonymousGraphCardClass.model_construct(representative=graph, multiplicity=3)
    deck = AnonymousGraphCardMultiset.model_construct(card_order=2, classes=(item,))
    with pytest.raises(OperationDomainValidationError, match="more edges"):
        anonymous_vertex_deck_edge_count(deck)
