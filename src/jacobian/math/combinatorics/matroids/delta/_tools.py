"""Finite delta-matroid operation declarations."""

from jacobian.catalog.models import (
    MathTool,
    MathTools,
    OperationDomainValidationError,
    OperationExample,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidFromFeasibleSetsRequest,
    DeltaMatroidRecognitionResult,
    DeltaMatroidTwistRequest,
    DeltaMatroidTwistResult,
    DeltaMatroidWidthRequest,
    DeltaMatroidWidthResult,
)
from jacobian.math.combinatorics.matroids.delta.operations import (
    from_feasible_sets,
    twist,
    width,
)
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


def _twist(request: DeltaMatroidTwistRequest) -> DeltaMatroidTwistResult:
    try:
        return twist(request)
    except DeltaMatroidAdmissionError as exc:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code=f"delta_matroid.{exc.reason}",
            message=str(exc),
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.source_not_valid",
            message=str(exc),
        ) from exc


def _width(request: DeltaMatroidWidthRequest) -> DeltaMatroidWidthResult:
    try:
        return width(request)
    except DeltaMatroidAdmissionError as exc:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code=f"delta_matroid.{exc.reason}",
            message=str(exc),
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.source_not_valid",
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
            OperationExample(
                name="two_element_delta_matroid",
                description="Recognize the complete feasible family on two labelled elements.",
                input={
                    "system": {
                        "ground": ["a", "b"],
                        "feasible": [[], [0], [1], [0, 1]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="delta_matroid.twist.compute",
        title="Twist a finite delta-matroid",
        description=(
            "Return the exact delta-matroid whose feasible sets are F symmetric "
            "difference X for each source feasible set F; the subset uses sorted "
            "ground indices and the source must satisfy symmetric exchange."
        ),
        request_type=DeltaMatroidTwistRequest,
        result_type=DeltaMatroidTwistResult,
        run=_twist,
        tags=("delta-matroid", "twist", "exact"),
        examples=(
            OperationExample(
                name="twist_by_first_element",
                description=(
                    "Twist {∅,{a},{b},{a,b}} by {a}; the subset must be sorted "
                    "ground indices."
                ),
                input={
                    "delta_matroid": {
                        "ground": ["a", "b"],
                        "feasible": [[], [0], [0, 1], [1]],
                    },
                    "subset": [0],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="delta_matroid.width.compute",
        title="Compute the width of a finite delta-matroid",
        description=(
            "Return max(|F|) - min(|F|) over the complete feasible family; the "
            "source must satisfy symmetric exchange."
        ),
        request_type=DeltaMatroidWidthRequest,
        result_type=DeltaMatroidWidthResult,
        run=_width,
        tags=("delta-matroid", "width", "exact"),
        examples=(
            OperationExample(
                name="uniform_two_element_family",
                description=(
                    "Compute the width of the complete two-element feasible family; "
                    "the source must satisfy symmetric exchange."
                ),
                input={
                    "delta_matroid": {
                        "ground": ["a", "b"],
                        "feasible": [[], [0], [0, 1], [1]],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
