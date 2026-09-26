"""Immutable declaration for anonymous vertex-deck degree reconstruction."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.decks._models import AnonymousGraphCardMultiset
from jacobian.math.graphs.decks.anonymous_vertex_degree_multiset._models import (
    AnonymousVertexDeckDegreeMultiset,
)
from jacobian.math.graphs.decks.anonymous_vertex_degree_multiset.operations import (
    anonymous_vertex_deck_degree_multiset,
)

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.deck.anonymous_vertex.degree_multiset.compute",
        title="Reconstruct the source degree multiset from an anonymous vertex deck",
        description=(
            "Given a complete anonymous vertex-deck multiset, first recover its "
            "source edge count, then compute each deleted vertex degree as "
            "source_edge_count minus the card edge count. Return the sorted "
            "degree multiset with the typed edge-count derivation. The operation "
            "rejects degrees outside the source axis, handshake failures, and "
            "nongraphical degree multisets. These necessary checks do not prove "
            "that an arbitrary card multiset is realizable as a vertex deck."
        ),
        request_type=AnonymousGraphCardMultiset,
        result_type=AnonymousVertexDeckDegreeMultiset,
        run=anonymous_vertex_deck_degree_multiset,
        tags=("graph", "deck", "anonymous", "vertex-deletion", "degree", "exact"),
        discovery_terms=(
            "anonymous vertex deck degree multiset",
            "reconstruct source degree sequence from graph cards",
            "vertex-deck degree sequence",
        ),
        examples=(
            OperationExample(
                name="path_three_vertex_deck",
                description=(
                    "Recover the source degree multiset (2, 1, 1) from the "
                    "anonymous vertex deck of a three-vertex path."
                ),
                input={
                    "card_order": 2,
                    "classes": [
                        {
                            "representative": {"vertices": ["v00", "v01"], "edges": []},
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
