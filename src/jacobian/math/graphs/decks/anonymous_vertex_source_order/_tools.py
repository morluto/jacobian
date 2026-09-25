"""Immutable declaration for anonymous vertex-deck source-order recovery."""

from typing import Any

from jacobian.catalog.models import MathTool
from jacobian.math.graphs.decks._models import AnonymousGraphCardMultiset
from jacobian.math.graphs.decks.anonymous_vertex_source_order._models import (
    AnonymousVertexDeckSourceOrder,
)
from jacobian.math.graphs.decks.anonymous_vertex_source_order.operations import (
    anonymous_vertex_deck_source_order,
)

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.deck.anonymous_vertex.source_order.compute",
        title="Recover source order from an anonymous vertex deck",
        description=(
            "For a complete multiset of vertex-deleted simple graph cards, infer "
            "the source order from its card order and total multiplicity. A "
            "nonempty n-vertex deck has n cards of order n-1; the order-zero "
            "graph has no cards and the order-one graph has one empty card. "
            "The result retains the exact anonymous card multiset. These "
            "necessary conditions do not prove that an arbitrary multiset is "
            "realizable as a graph deck."
        ),
        request_type=AnonymousGraphCardMultiset,
        result_type=AnonymousVertexDeckSourceOrder,
        run=anonymous_vertex_deck_source_order,
        tags=("graph", "deck", "anonymous", "vertex-deletion", "source-order", "exact"),
        discovery_terms=(
            "vertex deck source order",
            "number of vertices from a graph deck",
            "anonymous vertex-deck card count",
        ),
        examples=(),
    ),
)
