"""Operation declaration for monochromatic complete subhypergraph profiles."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.finite_structures.hypergraphs.monochromatic_complete_subhypergraph._models import (
    MonochromaticCompleteSubhypergraphProfile,
    MonochromaticCompleteSubhypergraphRequest,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.monochromatic_complete_subhypergraph.operations import (
    construct,
)


def _construct(
    request: MonochromaticCompleteSubhypergraphRequest,
) -> MonochromaticCompleteSubhypergraphProfile:
    return construct(
        request.coloring,
        request.source_uniformity,
        request.target_uniformity,
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="hypergraph.edge_colored.monochromatic_complete_subhypergraph.construct",
        title="Construct complete monochromatic uniform-subhypergraph profiles",
        description=(
            "Given a bounded indexed edge-coloured r-uniform finite hypergraph "
            "and s >= r, return every s-vertex set whose complete family of "
            "r-subsets occurs in the source with one common colour, together "
            "with aligned source-edge provenance. Missing source edges are not "
            "given an implicit colour."
        ),
        request_type=MonochromaticCompleteSubhypergraphRequest,
        result_type=MonochromaticCompleteSubhypergraphProfile,
        run=_construct,
        tags=("hypergraph", "coloring", "ramsey", "exact"),
        examples=(
            OperationExample(
                name="all_red_k4_3_uniform",
                description=(
                    "The four red triples of a complete 3-uniform K4 produce "
                    "one red target candidate."
                ),
                input={
                    "coloring": {
                        "hypergraph": {
                            "vertices": ["0", "1", "2", "3"],
                            "edges": [
                                ["012", ["0", "1", "2"]],
                                ["013", ["0", "1", "3"]],
                                ["023", ["0", "2", "3"]],
                                ["123", ["1", "2", "3"]],
                            ],
                        },
                        "color_count": 1,
                        "assignments": [
                            {"edge_id": edge_id, "color_index": 0}
                            for edge_id in ("012", "013", "023", "123")
                        ],
                    },
                    "source_uniformity": 3,
                    "target_uniformity": 4,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
