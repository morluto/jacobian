"""Immutable declarations for anonymous vertex-deck edge reconstruction."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
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
            "divided by n-2 is the implied edge count. Divisibility is "
            "necessary but does not establish deck realizability, so the result "
            "field is named realizing_edge_count: it is the edge count that any "
            "source graph realizing this deck must have, and this operation "
            "returns it whether or not such a graph exists. It is not a "
            "witnessed source graph and is not a realizability decision. Order "
            "zero and one use the empty-edge convention; order two is rejected "
            "because its deck does not determine whether the source edge exists."
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
        examples=(
            OperationExample(
                name="path_p3_edge_count",
                description=(
                    "The anonymous vertex deck of P3 has card edge total two; "
                    "dividing by n-2 gives realizing_edge_count two. The deck "
                    "is realizable, so this does match P3's two source edges, "
                    "but the operation does not itself establish realizability."
                ),
                input={
                    "card_order": 2,
                    "classes": [
                        {
                            "representative": {
                                "vertices": ["v00", "v01"],
                                "edges": [],
                            },
                            "multiplicity": "1",
                        },
                        {
                            "representative": {
                                "vertices": ["v00", "v01"],
                                "edges": [["v00", "v01"]],
                            },
                            "multiplicity": "2",
                        },
                    ],
                },
            ),
        ),
    ),
)
