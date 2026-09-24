"""Tests for exact source-bound edge-deletion deck operations."""

from __future__ import annotations

from itertools import combinations, permutations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.decks._models import (
    EdgeDeckRequest,
    EdgeDeletionFamily,
    SourceBoundEdgeCard,
    UnlabelledEdgeDeck,
    UnlabelledEdgeDeckRequest,
)
from jacobian.math.graphs.decks._tools import TOOLS, _run_edge_deleted
from jacobian.math.graphs.decks.operations import (
    edge_deletion_family,
    edge_unlabelled_deck,
    unlabelled_deck,
)
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


def test_serialized_unlabelled_quotient_does_not_replay_isomorphism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import networkx as nx

    quotient = unlabelled_deck(edge_deletion_family(_triangle()))

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("result validation must not replay graph isomorphism")

    monkeypatch.setattr(nx, "is_isomorphic", fail)

    assert type(quotient).model_validate_json(quotient.model_dump_json()) == quotient


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

    exact_tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "graph.deck.edge.unlabelled.compute"
    )
    exact_request = exact_tool.request_type.model_validate(exact_tool.examples[0].input)
    exact_result = exact_tool.run(exact_request)
    assert exact_result.card_count == 3
    assert exact_result.classes[0].multiplicity == 3
    assert exact_result.classes[0].deleted_edges == _triangle().edges


def _oracle_signature(graph: SimpleUndirectedGraph) -> tuple[int, ...]:
    """Independent adjacency-matrix permutation oracle for tiny graphs."""
    labels = graph.vertices
    adjacency = {frozenset(edge) for edge in graph.edges}
    candidates = []
    for order in permutations(labels):
        candidates.append(
            tuple(
                int(frozenset((order[i], order[j])) in adjacency)
                for i in range(len(order))
                for j in range(i + 1, len(order))
            )
        )
    return min(candidates, default=())


def test_edge_deck_classes_match_exhaustive_graph_permutation_oracle() -> None:
    # All 64 labelled simple graphs through order four are independently
    # classified by enumerating every vertex permutation of every card.
    for n in range(1, 5):
        vertices = tuple(f"v{i}" for i in range(n))
        possible_edges = tuple(combinations(vertices, 2))
        for mask in range(1 << len(possible_edges)):
            edges = tuple(
                edge for bit, edge in enumerate(possible_edges) if mask & (1 << bit)
            )
            graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)
            family = edge_deletion_family(graph)
            result = edge_unlabelled_deck(family)
            oracle_groups: dict[tuple[int, ...], list[int]] = {}
            for index, card in enumerate(family.cards):
                oracle_groups.setdefault(_oracle_signature(card.card), []).append(index)
            assert [item.card_indices for item in result.classes] == [
                tuple(indices) for indices in oracle_groups.values()
            ]
            assert [item.deleted_edges for item in result.classes] == [
                tuple(family.cards[index].deleted_edge for index in indices)
                for indices in oracle_groups.values()
            ]


def test_edge_deck_canonicalization_is_relabeling_invariant_and_round_trips() -> None:
    graph = _triangle()
    renamed = SimpleUndirectedGraph(
        vertices=("z", "x", "y"),
        edges=(("x", "y"), ("x", "z"), ("y", "z")),
    )
    first = edge_unlabelled_deck(edge_deletion_family(graph))
    second = edge_unlabelled_deck(edge_deletion_family(renamed))
    assert [
        (item.multiplicity, len(item.representative.edges)) for item in first.classes
    ] == [
        (item.multiplicity, len(item.representative.edges)) for item in second.classes
    ]
    decoded_family = EdgeDeletionFamily.model_validate_json(
        first.family.model_dump_json()
    )
    request = UnlabelledEdgeDeckRequest(deck=decoded_family)
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "graph.deck.edge.unlabelled.compute"
    )
    result = tool.run(request)
    decoded = UnlabelledEdgeDeck.model_validate_json(result.model_dump_json())
    assert decoded == result
    assert decoded.classes[0].card_indices == (0, 1, 2)
    assert decoded.classes[0].deleted_edges == graph.edges


def test_edge_unlabelled_catalog_operation_is_published() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "graph.deck.edge.unlabelled.compute"
    )
    assert tool.result_type is UnlabelledEdgeDeck
    assert (
        tool.run(tool.request_type.model_validate(tool.examples[0].input)).card_count
        == 3
    )
