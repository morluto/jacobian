"""Tests for exact source-bound edge-deletion deck operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.decks._models import (
    EdgeDeckRequest,
    EdgeDeletionFamily,
    SourceBoundEdgeCard,
)
from jacobian.math.graphs.decks._tools import TOOLS, _run_edge_deleted
from jacobian.math.graphs.decks.operations import edge_deletion_family, unlabelled_deck
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _triangle() -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=("a", "b", "c"),
        edges=(("a", "b"), ("a", "c"), ("b", "c")),
    )


def test_edge_cards_remove_exactly_the_bound_source_edge() -> None:
    result = edge_deletion_family(_triangle())

    assert [card.deleted_edge for card in result.cards] == list(result.source.edges)
    for card, deleted in zip(result.cards, result.source.edges, strict=True):
        assert card.card.edges == tuple(
            edge for edge in result.source.edges if edge != deleted
        )
        assert card.card.vertices == result.source.vertices


def test_serialized_family_composes_unchanged_into_unlabelled_quotient() -> None:
    family = edge_deletion_family(_triangle())
    decoded = EdgeDeletionFamily.model_validate_json(family.model_dump_json())

    quotient = unlabelled_deck(decoded)
    assert quotient.card_count == 3
    assert [item.multiplicity for item in quotient.classes] == [3]


def test_serialized_unlabelled_quotient_rejects_split_isomorphism_classes() -> None:
    family = edge_deletion_family(_triangle())
    quotient = unlabelled_deck(family)
    payload = quotient.model_dump(mode="json")
    payload["classes"] = [
        {
            "representative": family.cards[index].card.model_dump(mode="json"),
            "multiplicity": 1,
            "card_indices": [index],
        }
        for index in range(3)
    ]

    with pytest.raises(ValidationError):
        type(quotient).model_validate(payload)


def test_serialized_unlabelled_class_rejects_nonisomorphic_member() -> None:
    source = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(("a", "b"), ("b", "c"), ("c", "d")),
    )
    quotient = unlabelled_deck(edge_deletion_family(source))
    payload = quotient.model_dump(mode="json")
    payload["classes"][0]["card_indices"] = [0, 1]
    payload["classes"][0]["multiplicity"] = 2
    payload["classes"][1]["card_indices"] = [2]
    payload["classes"][1]["multiplicity"] = 1
    with pytest.raises(ValidationError):
        type(quotient).model_validate(payload)


def test_model_rejects_card_missing_another_source_edge() -> None:
    source = _triangle()
    with pytest.raises(ValidationError):
        EdgeDeletionFamily(
            source=source,
            cards=tuple(
                SourceBoundEdgeCard(
                    deleted_edge=deleted,
                    card=SimpleUndirectedGraph(vertices=source.vertices, edges=()),
                    retained_vertices=source.vertices,
                    retained_edge_count=0,
                )
                for deleted in source.edges
            ),
        )


def test_consumer_rejects_forged_native_family() -> None:
    source = _triangle()
    empty = SimpleUndirectedGraph(vertices=source.vertices, edges=())
    forged = EdgeDeletionFamily.model_construct(
        source=source,
        cards=tuple(
            SourceBoundEdgeCard.model_construct(
                deleted_edge=deleted,
                card=empty,
                retained_vertices=source.vertices,
                retained_edge_count=0,
            )
            for deleted in source.edges
        ),
    )

    with pytest.raises(OperationDomainValidationError) as exc_info:
        unlabelled_deck(forged)
    assert exc_info.value.errors()[0]["type"] == "graph_deck.card_deletion_relation"


def test_native_and_catalog_edge_family_parity() -> None:
    request = EdgeDeckRequest(graph=_triangle())
    assert _run_edge_deleted(request) == edge_deletion_family(request.graph)


def test_published_edge_deck_operations_have_executable_examples() -> None:
    edge_tool = next(
        tool for tool in TOOLS if tool.operation_id == "graph.deck.edge_deleted.compute"
    )
    edge_request = EdgeDeckRequest.model_validate(edge_tool.examples[0].input)
    family = edge_tool.run(edge_request)
    assert len(family.cards) == 3

    quotient_tool = next(
        tool for tool in TOOLS if tool.operation_id == "graph.deck.unlabelled.compute"
    )
    quotient_request = quotient_tool.request_type.model_validate(
        quotient_tool.examples[0].input
    )
    assert quotient_tool.run(quotient_request).card_count == 2
