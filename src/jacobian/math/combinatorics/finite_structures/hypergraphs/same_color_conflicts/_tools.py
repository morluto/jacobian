"""Same-colour union-conflict operation declaration."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.finite_structures.hypergraphs.same_color_conflicts._models import (
    SameColorConflictsRequest,
    SameColorConflictsResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.same_color_conflicts.operations import (
    construct,
)


def _construct(request: SameColorConflictsRequest) -> SameColorConflictsResult:
    return construct(request.coloring)


TOOLS: MathTools = (
    MathTool(
        operation_id="hypergraph.same_color_conflicts.construct",
        title="Construct complete same-colour union conflicts",
        description=(
            "Given an indexed edge colouring, return one conflict hyperedge per "
            "distinct union of two distinct equally coloured source edges, plus "
            "complete source-edge-pair provenance. A vertex subset is independent "
            "exactly when its induced source hypergraph is rainbow. Equal member "
            "sets with different source IDs remain distinct; duplicate unions "
            "share one conflict edge. Nonuniform and empty source edges are allowed. "
            "Two same-coloured empty source edges are rejected because their union "
            "is an empty hyperedge outside the carrier and independence_number. "
            "The conflict FiniteHypergraph composes with structural and independence "
            "operations (the independence consumer excludes empty hyperedges). "
            "Admits at most 65536 source pairs, 12000 distinct unions and 36000 "
            "union incidences on the existing 256-vertex carrier; no truncation."
        ),
        request_type=SameColorConflictsRequest,
        result_type=SameColorConflictsResult,
        run=_construct,
        tags=("hypergraph", "coloring", "rainbow", "conflict", "union"),
        examples=(
            OperationExample(
                name="repeated_triangle_edges",
                description=(
                    "Compute the same-colour union-conflict hypergraph of three "
                    "equally coloured triangle edges; assignments must cover every "
                    "source edge ID exactly once with a contiguous 0..color_count-1 "
                    "palette."
                ),
                input={
                    "coloring": {
                        "hypergraph": {
                            "vertices": ["a", "b", "c"],
                            "edges": [
                                ["ab", ["a", "b"]],
                                ["ac", ["a", "c"]],
                                ["bc", ["b", "c"]],
                            ],
                        },
                        "color_count": 1,
                        "assignments": [
                            {"edge_id": edge_id, "color_index": 0}
                            for edge_id in ("ab", "ac", "bc")
                        ],
                    }
                },
            ),
        ),
    ),
)
