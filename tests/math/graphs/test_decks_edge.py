"""Tests for exact source-bound edge-deletion deck operations."""

from __future__ import annotations

from itertools import combinations, permutations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.decks._models import (
    EdgeDeckRequest,
    EdgeDeletionFamily,
    SourceBoundEdgeCard,
    UnlabelledDeck,
    UnlabelledDeckRequest,
)
from jacobian.math.graphs.decks._tools import TOOLS, _run_edge_deleted
from jacobian.math.graphs.decks.operations import (
    edge_deletion_family,
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
    import jacobian.math.graphs.decks.operations as deck_operations

    quotient = unlabelled_deck(edge_deletion_family(_triangle()))

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("result validation must not replay canonicalization")

    monkeypatch.setattr(deck_operations, "_canonical_adjacency_signature", fail)

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
            result = unlabelled_deck(family)
            oracle_groups: dict[tuple[int, ...], list[int]] = {}
            for index, card in enumerate(family.cards):
                oracle_groups.setdefault(_oracle_signature(card.card), []).append(index)
            assert [item.card_indices for item in result.classes] == [
                tuple(indices) for indices in oracle_groups.values()
            ]
            assert [item.multiplicity for item in result.classes] == [
                len(indices) for indices in oracle_groups.values()
            ]
            assert [
                tuple(result.source.edges[index] for index in item.card_indices)
                for item in result.classes
            ] == [
                tuple(family.cards[index].deleted_edge for index in indices)
                for indices in oracle_groups.values()
            ]


def test_edge_deck_canonicalization_is_relabeling_invariant_and_round_trips() -> None:
    graph = _triangle()
    renamed = SimpleUndirectedGraph(
        vertices=("z", "x", "y"),
        edges=(("x", "y"), ("x", "z"), ("y", "z")),
    )
    family = edge_deletion_family(graph)
    first = unlabelled_deck(family)
    second = unlabelled_deck(edge_deletion_family(renamed))
    assert [
        (item.multiplicity, len(item.representative.edges)) for item in first.classes
    ] == [
        (item.multiplicity, len(item.representative.edges)) for item in second.classes
    ]
    decoded_family = EdgeDeletionFamily.model_validate_json(family.model_dump_json())
    request = UnlabelledDeckRequest(deck=decoded_family)
    tool = next(
        tool for tool in TOOLS if tool.operation_id == "graph.deck.unlabelled.compute"
    )
    result = tool.run(request)
    decoded = UnlabelledDeck.model_validate_json(result.model_dump_json())
    assert decoded == result
    assert decoded.classes[0].card_indices == (0, 1, 2)
    assert (
        tuple(decoded.source.edges[index] for index in decoded.classes[0].card_indices)
        == graph.edges
    )


def test_edge_deck_quotient_has_exactly_one_published_operation() -> None:
    """One postcondition, one operation: no competing edge-quotient IDs."""
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    quotients = sorted(
        tool.operation_id
        for tool in BUILTIN_TOOLS
        if tool.request_type.model_fields.get("deck") is not None
        and tool.request_type.model_fields["deck"].annotation is EdgeDeletionFamily
        and tool.result_type is UnlabelledDeck
    )
    assert quotients == ["graph.deck.unlabelled.compute"]


def test_edge_quotient_admits_exact_permutation_work_before_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The n^2 pairwise estimate must not admit 45 ten-vertex canonicalizations."""
    import jacobian.math.graphs.decks.operations as operations

    def fail(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("canonicalization ran before exact work admission")

    monkeypatch.setattr(operations, "_canonical_adjacency_signature", fail)
    vertices = tuple(f"v{i}" for i in range(10))
    edges = tuple(combinations(vertices, 2))
    family = edge_deletion_family(SimpleUndirectedGraph(vertices=vertices, edges=edges))
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        unlabelled_deck(family)
    assert exc_info.value.errors()[0]["type"] == "graph_deck.isomorphism_work_bound"
