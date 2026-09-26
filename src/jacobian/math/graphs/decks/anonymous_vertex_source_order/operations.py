"""Exact source-order reconstruction from anonymous vertex-deck cards."""

from __future__ import annotations

from math import comb

from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.decks._models import (
    MAX_ANONYMOUS_CARD_CLASSES,
    MAX_ANONYMOUS_CARD_RESULT_BYTES,
    MAX_UNLABELLED_DECK_VERTICES,
    AnonymousGraphCardClass,
    AnonymousGraphCardMultiset,
)
from jacobian.math.graphs.decks.anonymous_vertex_source_order._models import (
    AnonymousVertexDeckSourceOrder,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_ANONYMOUS_VERTEX_SOURCE_ORDER_RESULT_BYTES = MAX_ANONYMOUS_CARD_RESULT_BYTES + 128


def anonymous_vertex_deck_source_order(
    deck: AnonymousGraphCardMultiset,
) -> AnonymousVertexDeckSourceOrder:
    """Infer source order from card order and complete deck multiplicity.

    For a source with ``n > 0`` vertices, each vertex deletion has order
    ``n - 1`` and the multiset contains ``n`` cards. The unique order-zero
    graph has no cards; order one has one empty card. These necessary deck
    conditions determine source order, but do not establish that arbitrary
    cards are realizable as a graph's deck.
    """
    if type(deck) is not AnonymousGraphCardMultiset:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.vertex_source_order_carrier",
            message="deck must be an AnonymousGraphCardMultiset",
        )
    card_order = getattr(deck, "card_order", None)
    classes = getattr(deck, "classes", None)
    if type(card_order) is not int:
        raise OperationDomainValidationError(
            location=("deck", "card_order"),
            code="graph_deck.vertex_source_order_card_order",
            message="card order must be a strict bounded integer",
        )
    if not 0 <= card_order <= MAX_UNLABELLED_DECK_VERTICES:
        raise OperationResourceAdmissionError(
            location=("deck", "card_order"),
            code="graph_deck.vertex_source_order_order_bound",
            message="source-order reconstruction supports card orders through ten",
        )
    if type(classes) is not tuple:
        raise OperationDomainValidationError(
            location=("deck", "classes"),
            code="graph_deck.vertex_source_order_classes",
            message="deck classes must be an immutable tuple",
        )

    expected_count = 0 if card_order == 0 and not classes else card_order + 1
    if len(classes) > expected_count:
        raise OperationDomainValidationError(
            location=("deck", "classes"),
            code="graph_deck.vertex_source_order_card_count",
            message="a complete vertex deck cannot have more classes than cards",
        )
    pair_count = comb(card_order, 2)
    input_bytes = len(classes) * (64 + 16 * pair_count)
    output_bytes = input_bytes + 128
    if (
        len(classes) > MAX_ANONYMOUS_CARD_CLASSES
        or input_bytes > MAX_ANONYMOUS_CARD_RESULT_BYTES
        or output_bytes > MAX_ANONYMOUS_VERTEX_SOURCE_ORDER_RESULT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("deck", "classes"),
            code="graph_deck.vertex_source_order_output_bound",
            message="the retained anonymous deck exceeds the source-order result bound",
        )

    expected_vertices = tuple(f"v{index:02}" for index in range(card_order))
    card_count = 0
    for index, card_class in enumerate(classes):
        if type(card_class) is not AnonymousGraphCardClass:
            raise OperationDomainValidationError(
                location=("deck", "classes", index),
                code="graph_deck.vertex_source_order_class",
                message="deck entries must be canonical anonymous card classes",
            )
        multiplicity = getattr(card_class, "multiplicity", None)
        representative = getattr(card_class, "representative", None)
        edges = getattr(representative, "edges", None)
        vertices = getattr(representative, "vertices", None)
        if (
            type(multiplicity) is not int
            or not 1 <= multiplicity < 10**12
            or type(representative) is not SimpleUndirectedGraph
            or vertices != expected_vertices
            or type(edges) is not tuple
            or len(edges) > pair_count
            or any(
                type(edge) is not tuple
                or len(edge) != 2
                or type(edge[0]) is not str
                or type(edge[1]) is not str
                or edge[0] >= edge[1]
                or edge[0] not in expected_vertices
                or edge[1] not in expected_vertices
                for edge in edges
            )
        ):
            raise OperationDomainValidationError(
                location=("deck", "classes", index),
                code="graph_deck.vertex_source_order_class_shape",
                message="card classes must use bounded canonical representatives and positive multiplicities",
            )
        card_count += multiplicity
        if card_count > expected_count:
            raise OperationDomainValidationError(
                location=("deck", "classes", index, "multiplicity"),
                code="graph_deck.vertex_source_order_card_count",
                message="card multiplicities exceed the source order implied by card order",
            )

    source_order = card_count if card_order == 0 else card_order + 1
    if card_count != source_order:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.vertex_source_order_card_count",
            message="a complete vertex deck contains exactly source_order cards with multiplicity",
        )
    try:
        canonical_deck = AnonymousGraphCardMultiset.model_validate(
            deck.model_dump(mode="python")
        )
    except (ValidationError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.vertex_source_order_malformed",
            message="deck must use the canonical bounded anonymous-card carrier",
        ) from exc
    return AnonymousVertexDeckSourceOrder._from_kernel(canonical_deck, source_order)
