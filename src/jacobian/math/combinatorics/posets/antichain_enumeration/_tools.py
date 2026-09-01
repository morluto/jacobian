"""Typed declarations for the antichain enumeration operation."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
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
            OperationExample(
                name="chain_3",
                description="Antichains of size 1 in a 3-element chain (each element alone).",
                input={
                    "poset": {
                        "elements": ["a", "b", "c"],
                        "strict_order_pairs": [
                            {"lower": "a", "upper": "b"},
                            {"lower": "a", "upper": "c"},
                            {"lower": "b", "upper": "c"},
                        ],
                        "cover_relations": [
                            {"lower": "a", "upper": "b"},
                            {"lower": "b", "upper": "c"},
                        ],
                        "incomparable_pairs": [],
                        "minimal_elements": ["a"],
                        "maximal_elements": ["c"],
                        "graded": True,
                        "ranks": [
                            {"element": "a", "rank": 0},
                            {"element": "b", "rank": 1},
                            {"element": "c", "rank": 2},
                        ],
                        "poset_digest": "sha256:7505e31e11f07f0026eece8ce9621dd0dae51e613b8dfee93da02348bb80c95f",
                    },
                    "min_cardinality": 1,
                    "max_cardinality": 1,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
