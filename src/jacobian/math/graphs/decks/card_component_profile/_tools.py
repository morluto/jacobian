"""Tool declaration for cardwise connected-component profiles."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.decks.card_component_profile._models import (
    AnonymousDeckComponentProfile,
    AnonymousDeckComponentProfileRequest,
)
from jacobian.math.graphs.decks.card_component_profile.operations import (
    card_component_profile,
)


def _run(
    request: AnonymousDeckComponentProfileRequest,
) -> AnonymousDeckComponentProfile:
    return card_component_profile(request)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.deck.card_component_profile.compute",
        title="Profile connected component sizes across graph cards",
        description=(
            "For each isomorphism class in an anonymous same-order graph-card "
            "multiset, compute the sorted component vertex counts and return "
            "their exact histogram weighted by card multiplicity. Component "
            "sizes are relabelling invariants, so representatives are profiled "
            "directly under an aggregate 2000000-unit connectivity-work bound. "
            "This is a cardwise invariant and does not reconstruct the source "
            "graph's components."
        ),
        request_type=AnonymousDeckComponentProfileRequest,
        result_type=AnonymousDeckComponentProfile,
        run=_run,
        tags=("graph", "deck", "components", "invariant", "exact"),
        discovery_terms=(
            "graph deck component profile",
            "component sizes across graph cards",
            "cardwise connected components",
            "anonymous deck invariant",
        ),
        examples=(
            OperationExample(
                name="path_and_triangle_card_components",
                description=(
                    "Profile one path card and one triangle-plus-isolate card."
                ),
                input={
                    "deck": {
                        "card_order": 4,
                        "classes": [
                            {
                                "representative": {
                                    "vertices": ["v00", "v01", "v02", "v03"],
                                    "edges": [
                                        ["v00", "v01"],
                                        ["v00", "v02"],
                                        ["v01", "v02"],
                                    ],
                                },
                                "multiplicity": "1",
                            },
                            {
                                "representative": {
                                    "vertices": ["v00", "v01", "v02", "v03"],
                                    "edges": [
                                        ["v00", "v01"],
                                        ["v01", "v02"],
                                        ["v02", "v03"],
                                    ],
                                },
                                "multiplicity": "2",
                            },
                        ],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
