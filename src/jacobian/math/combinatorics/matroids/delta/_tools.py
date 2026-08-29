"""Finite delta-matroid operation declarations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationDomainValidationError
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidFromFeasibleSetsRequest,
    DeltaMatroidRecognitionResult,
)
from jacobian.math.combinatorics.matroids.delta.operations import from_feasible_sets
from jacobian.math.combinatorics.matroids.delta.values import DeltaMatroidAdmissionError


def _from_feasible_sets(
    request: DeltaMatroidFromFeasibleSetsRequest,
) -> DeltaMatroidRecognitionResult:
    """Recognize a complete feasible family as a finite delta-matroid."""

    try:
        return from_feasible_sets(request.system)
    except DeltaMatroidAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("system",),
            code=f"delta_matroid.{exc.reason}",
            message=str(exc),
        ) from exc


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
        run=_from_feasible_sets,
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
