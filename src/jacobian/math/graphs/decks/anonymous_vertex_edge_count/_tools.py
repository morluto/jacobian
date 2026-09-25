"""Immutable declarations for anonymous vertex-deck edge reconstruction."""

from typing import Any

from jacobian.catalog.models import MathTool
from jacobian.math.graphs.decks._models import AnonymousGraphCardMultiset
from jacobian.math.graphs.decks.anonymous_vertex_edge_count._models import (
    AnonymousVertexDeckEdgeCount,
)
from jacobian.math.graphs.decks.anonymous_vertex_edge_count.operations import (
    anonymous_vertex_deck_edge_count,
)

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.deck.anonymous_vertex.edge_count.compute",
        title="Reconstruct edge count from an anonymous vertex deck",
        description=(
            "Given an anonymous multiset of vertex-deleted simple graph cards, "
            "validate its bounded canonical card representation and compute the "
            "Kelly edge-count quotient. For source order n >= 3, each source "
            "edge appears in exactly n-2 cards, so the total card edge count "
            "divided by n-2 is the source edge count. Divisibility is necessary "
            "but does not establish deck realizability. Order zero and one use "
            "the empty-edge convention; order two is rejected because its deck "
            "does not determine whether the source edge exists."
        ),
        request_type=AnonymousGraphCardMultiset,
        result_type=AnonymousVertexDeckEdgeCount,
        run=anonymous_vertex_deck_edge_count,
        tags=("graph", "deck", "anonymous", "vertex-deletion", "edge-count", "exact"),
        discovery_terms=(
            "anonymous vertex deck edge count",
            "reconstruct edges from graph deck",
            "Kelly edge count identity",
        ),
        examples=(),
    ),
)
