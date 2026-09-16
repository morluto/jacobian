"""Exact native kernels for source-bound vertex-deletion families."""

from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.decks._models import (
    MAX_DECK_CARD_EDGES,
    MAX_DECK_VERTICES,
    SourceBoundVertexCard,
    VertexDeletionFamily,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = [
    "vertex_deletion_family",
    "verify_vertex_deletion_family",
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
    replays the Kelly counting identities: every source edge appears in
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
    edge_appearances = tuple(
        sum(1 for card in cards if edge in set(card.card.edges))
        for edge in source.edges
    )
    vertex_appearances = tuple(
        sum(1 for card in cards if vertex in set(card.card.vertices))
        for vertex in source.vertices
    )
    if order >= 2:
        expected_edge = order - 2
        if any(count != expected_edge for count in edge_appearances):
            raise OperationDomainValidationError(
                location=("graph",),
                code="graph_deck.edge_appearance_replay",
                message="every source edge must appear in exactly n-2 cards",
            )
    if any(count != order - 1 for count in vertex_appearances):
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph_deck.vertex_appearance_replay",
            message="every source vertex must appear in exactly n-1 card domains",
        )
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
