"""Tool declaration for anonymous graph-card multiset equality."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.decks.anonymous_equality._models import (
    AnonymousDeckEqualityRequest,
    AnonymousDeckEqualityResult,
)
from jacobian.math.graphs.decks.anonymous_equality.operations import (
    anonymous_deck_equality,
)


def _run(request: AnonymousDeckEqualityRequest) -> AnonymousDeckEqualityResult:
    return anonymous_deck_equality(request)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.deck.anonymous.equal.decide",
        title="Compare anonymous graph-card multisets",
        description=(
            "Decide exact equality of two finite multisets of same-order simple "
            "graphs up to independent card vertex relabelling. Equality means "
            "the same graph-isomorphism classes with the same multiplicities; "
            "it does not compare or reconstruct source graphs. Both sides are "
            "canonicalized under an aggregate 2000000-unit exact work bound."
        ),
        request_type=AnonymousDeckEqualityRequest,
        result_type=AnonymousDeckEqualityResult,
        run=_run,
        tags=("graph", "deck", "multiset", "isomorphism", "equality", "exact"),
        discovery_terms=(
            "anonymous deck equality",
            "multiset of graph cards",
            "independently relabelled cards",
            "graph deck comparison",
        ),
        examples=(
            OperationExample(
                name="same_single_card",
                description="Compare two one-card multisets containing isomorphic paths.",
                input={
                    "left": {
                        "card_order": 3,
                        "classes": [
                            {
                                "representative": {
                                    "vertices": ["v00", "v01", "v02"],
                                    "edges": [["v00", "v01"], ["v01", "v02"]],
                                },
                                "multiplicity": "1",
                            }
                        ],
                    },
                    "right": {
                        "card_order": 3,
                        "classes": [
                            {
                                "representative": {
                                    "vertices": ["v00", "v01", "v02"],
                                    "edges": [["v00", "v01"], ["v01", "v02"]],
                                },
                                "multiplicity": "1",
                            }
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
