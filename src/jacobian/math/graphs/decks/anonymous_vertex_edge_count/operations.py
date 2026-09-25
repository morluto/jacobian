"""Exact edge-count reconstruction for anonymous vertex-deck multisets."""

from __future__ import annotations

from math import comb, factorial

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.decks._models import (
    MAX_ANONYMOUS_CARD_CANONICALIZATION_WORK,
    AnonymousGraphCardClass,
    AnonymousGraphCardMultiset,
    _canonical_card_edges,
)
from jacobian.math.graphs.decks.anonymous_vertex_edge_count._models import (
    MAX_ANONYMOUS_VERTEX_DECK_ORDER,
    AnonymousVertexDeckEdgeCount,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _admit_card_class(
    item: object,
    *,
    index: int,
    card_order: int,
    pair_count: int,
) -> tuple[SimpleUndirectedGraph, int]:
    if type(item) is not AnonymousGraphCardClass:
        raise OperationDomainValidationError(
            location=("deck", "classes", index),
            code="graph_deck.anonymous_edge_count_class_carrier",
            message="deck entries must be AnonymousGraphCardClass values",
        )
    multiplicity = getattr(item, "multiplicity", None)
    graph = getattr(item, "representative", None)
    if (
        type(multiplicity) is not int
        or multiplicity < 1
        or multiplicity >= 10**12
        or type(graph) is not SimpleUndirectedGraph
        or type(graph.vertices) is not tuple
        or type(graph.edges) is not tuple
    ):
        raise OperationDomainValidationError(
            location=("deck", "classes", index),
            code="graph_deck.anonymous_edge_count_class_shape",
            message="deck classes must contain bounded graph representatives and positive multiplicities",
        )
    expected_vertices = tuple(f"v{i:02d}" for i in range(card_order))
    invalid_edges = any(
        type(edge) is not tuple
        or len(edge) != 2
        or type(edge[0]) is not str
        or type(edge[1]) is not str
        or edge[0] >= edge[1]
        or edge[0] not in expected_vertices
        or edge[1] not in expected_vertices
        for edge in graph.edges
    )
    if (
        graph.vertices != expected_vertices
        or len(graph.edges) > pair_count
        or invalid_edges
        or len(set(graph.edges)) != len(graph.edges)
        or tuple(sorted(graph.edges)) != graph.edges
    ):
        raise OperationDomainValidationError(
            location=("deck", "classes", index),
            code="graph_deck.anonymous_edge_count_representative",
            message="deck representatives must use canonical labels and valid ordered edges",
        )
    return graph, multiplicity


def _admit_deck_structure(
    deck: AnonymousGraphCardMultiset,
) -> tuple[int, tuple[AnonymousGraphCardClass, ...], int]:
    if type(deck) is not AnonymousGraphCardMultiset:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.anonymous_edge_count_carrier",
            message="deck must be an AnonymousGraphCardMultiset",
        )
    card_order = getattr(deck, "card_order", None)
    classes = getattr(deck, "classes", None)
    if (
        type(card_order) is not int
        or not 0 <= card_order <= MAX_ANONYMOUS_VERTEX_DECK_ORDER - 1
    ):
        raise OperationResourceAdmissionError(
            location=("deck", "card_order"),
            code="graph_deck.anonymous_edge_count_order_bound",
            message="exact anonymous vertex-deck edge count supports card orders zero through seven",
        )
    if type(classes) is not tuple:
        raise OperationDomainValidationError(
            location=("deck", "classes"),
            code="graph_deck.anonymous_edge_count_classes",
            message="deck classes must be an immutable tuple",
        )

    pair_count = comb(card_order, 2)
    if len(classes) > 8 or len(classes) * (64 + 16 * pair_count) > 1_000_000:
        raise OperationResourceAdmissionError(
            location=("deck", "classes"),
            code="graph_deck.anonymous_edge_count_result_bound",
            message="anonymous deck classes exceed the admitted result bound",
        )
    work = len(classes) * factorial(card_order) * (card_order + 2 * max(1, pair_count))
    if work > MAX_ANONYMOUS_CARD_CANONICALIZATION_WORK:
        raise OperationResourceAdmissionError(
            location=("deck", "classes"),
            code="graph_deck.anonymous_edge_count_canonicalization_bound",
            message="canonical card validation exceeds the admitted work bound",
        )

    card_count = 0
    card_edge_total = 0
    previous_key: tuple[tuple[str, str], ...] | None = None
    for index, item in enumerate(classes):
        graph, multiplicity = _admit_card_class(
            item, index=index, card_order=card_order, pair_count=pair_count
        )
        if previous_key is not None and graph.edges <= previous_key:
            raise OperationDomainValidationError(
                location=("deck", "classes", index),
                code="graph_deck.anonymous_edge_count_class_order",
                message="deck classes must be unique and lexicographically ordered",
            )
        previous_key = graph.edges
        card_count += multiplicity
        card_edge_total += len(graph.edges) * multiplicity
        if card_count > MAX_ANONYMOUS_VERTEX_DECK_ORDER:
            raise OperationResourceAdmissionError(
                location=("deck", "classes", index, "multiplicity"),
                code="graph_deck.anonymous_edge_count_card_bound",
                message="vertex-deck multiplicities exceed the source-order bound",
            )

    if card_order == 0:
        if card_count not in (0, 1) or card_edge_total != 0:
            raise OperationDomainValidationError(
                location=("deck",),
                code="graph_deck.anonymous_edge_count_small_deck",
                message="card order zero supports only the empty order-zero deck or one order-one card",
            )
        source_order = card_count
    else:
        source_order = card_order + 1
        if card_count != source_order:
            raise OperationDomainValidationError(
                location=("deck",),
                code="graph_deck.anonymous_edge_count_card_count",
                message="a vertex deck must contain exactly source_order cards with multiplicity",
            )
        if source_order == 2:
            raise OperationDomainValidationError(
                location=("deck",),
                code="graph_deck.anonymous_edge_count_order_two",
                message="order-two vertex decks do not determine the source edge count",
            )
    return source_order, classes, card_edge_total


def _require_canonical_classes(
    classes: tuple[AnonymousGraphCardClass, ...],
) -> None:
    for index, item in enumerate(classes):
        graph = item.representative
        if _canonical_card_edges(graph.vertices, graph.edges) != graph.edges:
            raise OperationDomainValidationError(
                location=("deck", "classes", index),
                code="graph_deck.anonymous_edge_count_canonicality",
                message="each deck representative must be the least fixed-axis isomorphism encoding",
            )


def anonymous_vertex_deck_edge_count(
    deck: AnonymousGraphCardMultiset,
) -> AnonymousVertexDeckEdgeCount:
    """Recover the source edge count from a realizable anonymous vertex deck.

    For each edge of an n-vertex graph, exactly n-2 deleted-vertex cards retain
    that edge. Thus the sum of all card edge counts is (n-2)|E|. The operation
    checks the deck's bounded canonical representation and card multiplicity,
    then applies this necessary identity; divisibility alone does not prove
    that arbitrary cards form a realizable deck.
    """
    # Source-order, multiplicity, shape and work bounds pass before factorial
    # canonical-form validation.
    source_order, classes, card_edge_total = _admit_deck_structure(deck)
    _require_canonical_classes(classes)
    if source_order < 3:
        return AnonymousVertexDeckEdgeCount._from_kernel(
            deck=deck,
            source_order=source_order,
            card_edge_total=0,
            overcount_divisor=None,
            source_edge_count=0,
        )

    divisor = source_order - 2
    if card_edge_total % divisor:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.anonymous_edge_count_nondivisible",
            message="card edge total is not divisible by source order minus two",
        )
    return AnonymousVertexDeckEdgeCount._from_kernel(
        deck=deck,
        source_order=source_order,
        card_edge_total=card_edge_total,
        overcount_divisor=divisor,
        source_edge_count=card_edge_total // divisor,
    )
