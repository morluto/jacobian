"""Maximal finite-poset chain enumeration declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.posets.core._maximal_chains import (
    MaximalChainEnumerationResult,
    enumerate_maximal_chains,
)
from jacobian.math.combinatorics.posets.core._models import PosetRequest

MAXIMAL_CHAIN_ENUMERATION_OPERATION = MathTool(
    operation_id="poset.maximal_chains.enumerate",
    title="Enumerate all maximal chains of a finite poset",
    description=(
        "Enumerate every source-bound maximal chain from the Hasse cover relation, "
        "with endpoints, cardinality histogram, and the single empty chain for the empty poset."
    ),
    request_type=PosetRequest,
    result_type=MaximalChainEnumerationResult,
    run=enumerate_maximal_chains,
    tags=("poset", "chain", "maximal", "enumerate", "complete"),
    examples=(
        OperationExample(
            name="diamond",
            description="Enumerate the two maximal chains of the four-element diamond.",
            input={
                "poset": {
                    "elements": ["0", "1", "a", "b"],
                    "strict_order_pairs": [
                        {"lower": "0", "upper": "1"},
                        {"lower": "0", "upper": "a"},
                        {"lower": "0", "upper": "b"},
                        {"lower": "a", "upper": "1"},
                        {"lower": "b", "upper": "1"},
                    ],
                    "cover_relations": [
                        {"lower": "0", "upper": "a"},
                        {"lower": "0", "upper": "b"},
                        {"lower": "a", "upper": "1"},
                        {"lower": "b", "upper": "1"},
                    ],
                    "incomparable_pairs": [{"left": "a", "right": "b"}],
                    "minimal_elements": ["0"],
                    "maximal_elements": ["1"],
                    "graded": True,
                    "ranks": [
                        {"element": "0", "rank": 0},
                        {"element": "1", "rank": 2},
                        {"element": "a", "rank": 1},
                        {"element": "b", "rank": 1},
                    ],
                    "poset_digest": "sha256:55e795bf7924508b0aa0efe1d0cf32371858ff8b122c890d3eba357dfe2a3374",
                }
            },
        ),
    ),
)
