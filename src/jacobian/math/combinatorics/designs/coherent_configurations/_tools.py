"""Coherent-configuration operation declarations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.combinatorics.designs.coherent_configurations._models import (
    CoherentConfigurationAnalyzeRequest,
    CoherentConfigurationAnalyzeResult,
)
from jacobian.math.combinatorics.designs.coherent_configurations._operations import (
    compute_analyze,
)

TOOLS: MathTools = (
    MathTool(
        operation_id="coherent_configuration.analyze.compute",
        title="Analyze a complete finite coherent configuration",
        description=(
            "Check a complete labelled partition of ordered point pairs for the "
            "coherent-configuration axioms. Returns its exact fibres, transpose "
            "map, and intersection tensor, or the first concrete obstruction."
        ),
        request_type=CoherentConfigurationAnalyzeRequest,
        result_type=CoherentConfigurationAnalyzeResult,
        run=compute_analyze,
        tags=(
            "algebraic-combinatorics",
            "coherent-configuration",
            "association-scheme",
            "exact",
            "bounded",
        ),
        examples=(
            example(
                "complete_graph_rank_two",
                "Analyze the two-relation coherent configuration of K3.",
                {
                    "configuration": {
                        "points": ["a", "b", "c"],
                        "relation_ids": ["diagonal", "edge"],
                        "relation_matrix": [
                            ["diagonal", "edge", "edge"],
                            ["edge", "diagonal", "edge"],
                            ["edge", "edge", "diagonal"],
                        ],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
