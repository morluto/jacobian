"""Typed declarations for the antichain enumeration operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.combinatorics.posets.antichain_enumeration._models import (
    AntichainEnumerationRequest,
    AntichainEnumerationResult,
)
from jacobian.math.combinatorics.posets.antichain_enumeration.operations import (
    enumerate_antichains,
)


def _enumerate(request: AntichainEnumerationRequest) -> AntichainEnumerationResult:
    return enumerate_antichains(
        request.poset,
        request.min_cardinality,
        request.max_cardinality,
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="poset.antichain.enumerate",
        title="Enumerate antichains of a finite poset in requested cardinalities",
        description=(
            "For one bounded finite poset and a cardinality range, return every "
            "antichain (set of pairwise incomparable elements) of those sizes "
            "exactly once."
        ),
        request_type=AntichainEnumerationRequest,
        result_type=AntichainEnumerationResult,
        run=_enumerate,
        tags=("poset", "antichain", "enumeration", "exact"),
        examples=(
            example(
                "chain_3",
                "Antichains of size 1 in a 3-element chain (each element alone).",
                {
                    "poset": {
                        "elements": ["a", "b", "c"],
                        "strict_order_pairs": [["a", "b"], ["a", "c"], ["b", "c"]],
                        "cover_relations": [["a", "b"], ["b", "c"]],
                        "incomparable_pairs": [],
                        "minimal_elements": ["a"],
                        "maximal_elements": ["c"],
                        "graded": True,
                        "ranks": [{"element": "a", "rank": 0}, {"element": "b", "rank": 1}, {"element": "c", "rank": 2}],
                        "poset_digest": "0000000000000000000000000000000000000000000000000000000000000000",
                    },
                    "min_cardinality": 1,
                    "max_cardinality": 1,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
