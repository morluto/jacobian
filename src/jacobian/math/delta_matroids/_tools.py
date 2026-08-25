"""Finite delta-matroid operation declarations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.delta_matroids._models import (
    DeltaMatroidFromFeasibleSetsRequest,
    DeltaMatroidRecognitionResult,
)
from jacobian.math.delta_matroids._operations import compute_from_feasible_sets

TOOLS: MathTools = (
    MathTool(
        operation_id="delta_matroid.from_feasible_sets.compute",
        title="Recognize a finite delta-matroid from a complete feasible family",
        description=(
            "Exhaust the symmetric-exchange axiom for one complete bounded "
            "feasible-set family. Return its canonical finite delta-matroid "
            "value or the first deterministic exchange obstruction."
        ),
        request_type=DeltaMatroidFromFeasibleSetsRequest,
        result_type=DeltaMatroidRecognitionResult,
        run=compute_from_feasible_sets,
        tags=("delta-matroid", "symmetric-exchange", "exact"),
        examples=(
            example(
                "two_element_delta_matroid",
                "Recognize the complete feasible family on two labelled elements.",
                {
                    "system": {
                        "ground": ["a", "b"],
                        "feasible": [[], [0], [1], [0, 1]],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
