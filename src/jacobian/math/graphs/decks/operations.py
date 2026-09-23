"""Exact native kernels for source-bound vertex-deletion families."""

from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.decks._models import (
    MAX_DECK_CARD_EDGES,
    MAX_DECK_VERTICES,
    MAX_EDGE_DECK_EDGES,
    MAX_UNLABELLED_DECK_ISOMORPHISM_WORK,
    MAX_UNLABELLED_DECK_VERTICES,
    EdgeDeletionFamily,
    SourceBoundEdgeCard,
    SourceBoundVertexCard,
    UnlabelledDeck,
    UnlabelledDeckClass,
    VertexDeletionFamily,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = [
    "edge_deletion_family",
    "unlabelled_deck",
    "verify_edge_deletion_family",
    "verify_vertex_deletion_family",
    "vertex_deletion_family",
]


def _admit_deck_graph(graph: SimpleUndirectedGraph) -> SimpleUndirectedGraph:
    """Shared deck admission for native and catalog callers.

    Bounds the source order and the aggregate card-edge output before any
    card materialization.
    """
    if type(graph) is not SimpleUndirectedGraph:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph_deck.graph_carrier",
            message="vertex decks require a SimpleUndirectedGraph value",
        )
    order = len(graph.vertices)
    if order > MAX_DECK_VERTICES:
        raise OperationResourceAdmissionError(
            location=("graph",),
            code="graph_deck.vertex_bound",
            message=(
                "vertex-deletion decks support at most "
                f"{MAX_DECK_VERTICES} source vertices"
            ),
        )
    edge_count = len(graph.edges)
    if order >= 2 and (order - 2) * edge_count > MAX_DECK_CARD_EDGES:
        raise OperationResourceAdmissionError(
            location=("graph",),
            code="graph_deck.card_edge_bound",
            message="aggregate card edges exceed the exact output bound",
        )
    return graph


def _admit_edge_deck_graph(graph: SimpleUndirectedGraph) -> SimpleUndirectedGraph:
    if type(graph) is not SimpleUndirectedGraph:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph_deck.graph_carrier",
            message="edge decks require a SimpleUndirectedGraph value",
        )
    edge_count = len(graph.edges)
    if edge_count * max(edge_count - 1, 0) > MAX_EDGE_DECK_EDGES:
        raise OperationResourceAdmissionError(
            location=("graph",),
            code="graph_deck.edge_card_edge_bound",
            message="aggregate edge-card output exceeds the exact bound",
        )
    return graph


def edge_deletion_family(graph: SimpleUndirectedGraph) -> EdgeDeletionFamily:
    """Return every source-edge-deleted card, retaining isolated vertices."""
    source = _admit_edge_deck_graph(graph)
    cards: list[SourceBoundEdgeCard] = []
    for deleted in source.edges:
        card_edges = tuple(edge for edge in source.edges if edge != deleted)
        cards.append(
            SourceBoundEdgeCard.model_construct(
                deleted_edge=deleted,
                card=SimpleUndirectedGraph(vertices=source.vertices, edges=card_edges),
                retained_vertices=source.vertices,
                retained_edge_count=len(card_edges),
            )
        )
    return EdgeDeletionFamily._from_kernel(source=source, cards=tuple(cards))


def verify_edge_deletion_family(claim: EdgeDeletionFamily) -> bool:
    if type(claim) is not EdgeDeletionFamily:
        return False
    return edge_deletion_family(claim.source) == claim


def _isomorphic(left: SimpleUndirectedGraph, right: SimpleUndirectedGraph) -> bool:
    import networkx as nx

    first: nx.Graph[str] = nx.Graph()
    first.add_nodes_from(left.vertices)
    first.add_edges_from(left.edges)
    second: nx.Graph[str] = nx.Graph()
    second.add_nodes_from(right.vertices)
    second.add_edges_from(right.edges)
    return nx.is_isomorphic(first, second)


def _admit_edge_deletion_family(family: EdgeDeletionFamily) -> EdgeDeletionFamily:
    """Re-establish the exact source-minus-edge relation at a consumer boundary."""
    if type(family) is not EdgeDeletionFamily:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.family_carrier",
            message="deck must be an EdgeDeletionFamily",
        )
    source = family.source
    _admit_edge_deck_graph(source)
    if type(family.cards) is not tuple or len(family.cards) != len(source.edges):
        raise OperationDomainValidationError(
            location=("deck", "cards"),
            code="graph_deck.family_card_count",
            message="edge deletion family must contain one card per source edge",
        )
    for expected_edge, card in zip(source.edges, family.cards, strict=True):
        if type(card) is not SourceBoundEdgeCard:
            raise OperationDomainValidationError(
                location=("deck", "cards"),
                code="graph_deck.card_carrier",
                message="edge deletion family cards must use the canonical card type",
            )
        if (
            card.deleted_edge != expected_edge
            or type(card.card) is not SimpleUndirectedGraph
            or card.retained_vertices != source.vertices
            or card.card.vertices != source.vertices
            or card.card.edges
            != tuple(edge for edge in source.edges if edge != expected_edge)
            or card.retained_edge_count != len(card.card.edges)
        ):
            raise OperationDomainValidationError(
                location=("deck", "cards"),
                code="graph_deck.card_deletion_relation",
                message="each edge card must equal the source graph with its bound edge removed",
            )
    return family


def unlabelled_deck(family: EdgeDeletionFamily) -> UnlabelledDeck:
    """Quotient an edge deck into exact unlabeled isomorphism classes."""
    family = _admit_edge_deletion_family(family)
    n = len(family.source.vertices)
    if n > MAX_UNLABELLED_DECK_VERTICES:
        raise OperationResourceAdmissionError(
            location=("deck",),
            code="graph_deck.isomorphism_vertex_bound",
            message="unlabelled quotient exceeds the isomorphism envelope",
        )
    if len(family.cards) * max(n, 1) * max(n, 1) > MAX_UNLABELLED_DECK_ISOMORPHISM_WORK:
        raise OperationResourceAdmissionError(
            location=("deck",),
            code="graph_deck.isomorphism_work_bound",
            message="unlabelled quotient exceeds the isomorphism work envelope",
        )
    classes: list[UnlabelledDeckClass] = []
    for index, card in enumerate(family.cards):
        for class_index, item in enumerate(classes):
            if _isomorphic(card.card, item.representative):
                classes[class_index] = item.model_copy(
                    update={
                        "multiplicity": item.multiplicity + 1,
                        "card_indices": (*item.card_indices, index),
                    }
                )
                break
        else:
            classes.append(
                UnlabelledDeckClass.model_construct(
                    representative=card.card, multiplicity=1, card_indices=(index,)
                )
            )
    return UnlabelledDeck._from_kernel(
        source=family.source, classes=tuple(classes), card_count=len(family.cards)
    )


def _delete_vertex(
    graph: SimpleUndirectedGraph, deleted: str
) -> tuple[SimpleUndirectedGraph, int, int]:
    """Return the card of ``deleted`` plus retained/deleted edge counts."""
    retained = tuple(vertex for vertex in graph.vertices if vertex != deleted)
    retained_set = set(retained)
    card_edges = tuple(
        edge
        for edge in graph.edges
        if edge[0] in retained_set and edge[1] in retained_set
    )
    card = SimpleUndirectedGraph(vertices=retained, edges=card_edges)
    return card, len(card_edges), len(graph.edges) - len(card_edges)


def vertex_deletion_family(graph: SimpleUndirectedGraph) -> VertexDeletionFamily:
    """Build the complete source-bound vertex-deletion family of ``graph``.

    Iterates the exact source vertex domain once, deletes each vertex
    directly, binds every retained vertex and edge to the source, then
    uses the Kelly counting identities: every source edge appears in
    exactly ``n-2`` cards (for ``n >= 2``) and every source vertex in
    exactly ``n-1`` card domains.
    """
    source = _admit_deck_graph(graph)
    order = len(source.vertices)
    cards: list[SourceBoundVertexCard] = []
    for deleted in source.vertices:
        card, retained_count, deleted_count = _delete_vertex(source, deleted)
        cards.append(
            SourceBoundVertexCard.model_construct(
                deleted_vertex=deleted,
                card=card,
                retained_vertices=card.vertices,
                retained_edge_count=retained_count,
                deleted_edge_count=deleted_count,
            )
        )
    edge_appearances = (max(order - 2, 0),) * len(source.edges)
    vertex_appearances = (max(order - 1, 0),) * order
    return VertexDeletionFamily._from_kernel(
        source,
        tuple(cards),
        edge_appearances,
        vertex_appearances,
    )


def verify_vertex_deletion_family(claim: VertexDeletionFamily) -> bool:
    """Replay deletion and counting identities for a serialized family."""
    try:
        return vertex_deletion_family(claim.source) == claim
    except (OperationDomainValidationError, TypeError, ValueError):
        return False
