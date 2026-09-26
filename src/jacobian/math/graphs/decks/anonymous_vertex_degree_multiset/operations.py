"""Exact source degree-multiset reconstruction from anonymous vertex decks."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.decks._models import AnonymousGraphCardMultiset
from jacobian.math.graphs.decks.anonymous_vertex_degree_multiset._models import (
    AnonymousVertexDeckDegreeMultiset,
)
from jacobian.math.graphs.decks.anonymous_vertex_edge_count.operations import (
    anonymous_vertex_deck_edge_count,
)
from jacobian.math.graphs.realization._models import DegreeSequence
from jacobian.math.graphs.realization.operations import degree_sequence_profile


def anonymous_vertex_deck_degree_multiset(
    deck: AnonymousGraphCardMultiset,
) -> AnonymousVertexDeckDegreeMultiset:
    """Recover the source degree multiset through the anonymous edge count.

    For each deleted-vertex card ``G-v``, ``deg_G(v)=|E(G)|-|E(G-v)|``.
    The edge-count operation authenticates the bounded anonymous deck and
    recovers ``|E(G)|`` first. This operation then checks that every resulting
    degree is in range, that the sum is ``2|E(G)|``, and that the multiset is
    graphical. These necessary checks do not prove that the cards themselves
    form a realizable deck.
    """
    edge_count = anonymous_vertex_deck_edge_count(deck)
    order = edge_count.source_order
    source_edges = edge_count.source_edge_count
    degrees = tuple(
        sorted(
            (
                source_edges - len(card_class.representative.edges)
                for card_class in edge_count.deck.classes
                for _ in range(card_class.multiplicity)
            ),
            reverse=True,
        )
    )
    if len(degrees) != order or any(
        degree < 0 or degree >= order for degree in degrees
    ):
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.anonymous_degree_range",
            message="card edge counts imply degrees outside the source graph's vertex axis",
        )
    if sum(degrees) != 2 * source_edges:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.anonymous_degree_handshake",
            message="reconstructed degrees violate the handshake identity",
        )
    if (
        order
        and not degree_sequence_profile(DegreeSequence(degrees=degrees)).is_graphical
    ):
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.anonymous_degree_nongraphical",
            message="card edge counts imply a nongraphical source degree multiset",
        )
    return AnonymousVertexDeckDegreeMultiset.model_construct(
        edge_count=edge_count,
        degrees=degrees,
    )
