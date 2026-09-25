"""Finite delta-matroid operation declarations."""

from jacobian.catalog.models import (
    MathTool,
    MathTools,
    OperationDomainValidationError,
    OperationExample,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidExtremalMatroidResult,
    DeltaMatroidFromFeasibleSetsRequest,
    DeltaMatroidLowerMatroidRequest,
    DeltaMatroidRecognitionResult,
    DeltaMatroidTwistRequest,
    DeltaMatroidUpperMatroidRequest,
    DeltaMatroidWidthRequest,
    DeltaMatroidWidthResult,
)
from jacobian.math.combinatorics.matroids.delta.extra import (
    BinaryMatrixRequest,
    BinaryMatrixResult,
    DeltaMatroidDualRequest,
    DeltaMatroidMinorRequest,
)
from jacobian.math.combinatorics.matroids.delta.extra_ops import binary, dual, minor
from jacobian.math.combinatorics.matroids.delta.operations import (
    from_feasible_sets,
    lower_matroid,
    twist,
    upper_matroid,
    width,
)
from jacobian.math.combinatorics.matroids.delta.values import (
    DeltaMatroidAdmissionError,
    FiniteDeltaMatroid,
)


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


def _twist(request: DeltaMatroidTwistRequest) -> FiniteDeltaMatroid:
    try:
        return twist(request.delta_matroid, request.subset)
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


def _extra_domain(
    location: tuple[str, ...], code: str, exc: Exception
) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=location, code=code, message=str(exc)
    )


def _run_dual(request: DeltaMatroidDualRequest) -> FiniteDeltaMatroid:
    try:
        return dual(request.delta_matroid)
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        raise
    except (TypeError, ValueError, IndexError) as exc:
        raise _extra_domain(
            ("delta_matroid",), "delta_matroid.source_not_valid", exc
        ) from exc


def _run_minor(request: DeltaMatroidMinorRequest) -> FiniteDeltaMatroid:
    try:
        return minor(request.delta_matroid, request.delete, request.contract)
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        raise
    except (TypeError, ValueError, IndexError) as exc:
        raise _extra_domain(
            ("delta_matroid",), "delta_matroid.source_not_valid", exc
        ) from exc


def _run_binary(request: BinaryMatrixRequest) -> BinaryMatrixResult:
    try:
        return binary(request.matrix)
    except OperationResourceAdmissionError:
        raise
    except (TypeError, ValueError, IndexError) as exc:
        raise _extra_domain(("matrix",), "delta_matroid.binary_invalid", exc) from exc


def _width(request: DeltaMatroidWidthRequest) -> DeltaMatroidWidthResult:
    try:
        return DeltaMatroidWidthResult._from_kernel(
            request.delta_matroid, width(request.delta_matroid)
        )
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


def _lower_matroid(
    request: DeltaMatroidLowerMatroidRequest,
) -> DeltaMatroidExtremalMatroidResult:
    return lower_matroid(request.delta_matroid)


def _upper_matroid(
    request: DeltaMatroidUpperMatroidRequest,
) -> DeltaMatroidExtremalMatroidResult:
    return upper_matroid(request.delta_matroid)


TOOLS: MathTools = (  # noqa: RUF005
    MathTool(
        operation_id="delta_matroid.lower_matroid.compute",
        title="Compute the lower matroid of a finite delta-matroid",
        description=(
            "Return the matroid whose complete basis family consists of every "
            "minimum-cardinality feasible set. Retain the exact source ground "
            "axis and map each output basis to its source feasible-row index. "
            "Source exchange is replayed; output ground, family, membership, "
            "label, and basis-exchange work are admitted before construction."
        ),
        request_type=DeltaMatroidLowerMatroidRequest,
        result_type=DeltaMatroidExtremalMatroidResult,
        run=_lower_matroid,
        tags=("delta-matroid", "lower-matroid", "exact"),
        examples=(
            OperationExample(
                name="lower_matroid_of_two_element_delta",
                description="The unique minimum-cardinality feasible set is ∅.",
                input={
                    "delta_matroid": {
                        "ground": ["a", "b"],
                        "feasible": [[], [0], [0, 1], [1]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="delta_matroid.upper_matroid.compute",
        title="Compute the upper matroid of a finite delta-matroid",
        description=(
            "Return the matroid whose complete basis family consists of every "
            "maximum-cardinality feasible set. Retain the exact source ground "
            "axis and map each output basis to its source feasible-row index. "
            "Source exchange is replayed; output ground, family, membership, "
            "label, and basis-exchange work are admitted before construction."
        ),
        request_type=DeltaMatroidUpperMatroidRequest,
        result_type=DeltaMatroidExtremalMatroidResult,
        run=_upper_matroid,
        tags=("delta-matroid", "upper-matroid", "exact"),
        examples=(
            OperationExample(
                name="upper_matroid_of_two_element_delta",
                description="The unique maximum feasible set is {a,b}.",
                input={
                    "delta_matroid": {
                        "ground": ["a", "b"],
                        "feasible": [[], [0], [0, 1], [1]],
                    }
                },
            ),
        ),
    ),
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
        operation_id="delta_matroid.width.compute",
        title="Compute the width of a finite delta-matroid",
        description=(
            "Return the exact width max |F| - min |F| of one canonical finite "
            "delta-matroid over its complete feasible family. The carrier is "
            "reconstructed before measuring so a forged value with malformed "
            "rows is a domain reject, not a number."
        ),
        request_type=DeltaMatroidWidthRequest,
        result_type=DeltaMatroidWidthResult,
        run=_width,
        tags=("delta-matroid", "width", "exact"),
        discovery_terms=("delta-matroid width", "feasible size range"),
        examples=(
            OperationExample(
                name="width_two_family",
                description="Width of {∅,{a},{b},{a,b}} is 2 - 0 = 2.",
                input={
                    "delta_matroid": {
                        "ground": ["a", "b"],
                        "feasible": [[], [0], [0, 1], [1]],
                    },
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
            "ground indices. Source and output families are admitted at 16,384 "
            "memberships, 2,048 UTF-8 label bytes, and 250,000 symmetric-exchange "
            "candidate checks."
        ),
        request_type=DeltaMatroidTwistRequest,
        result_type=FiniteDeltaMatroid,
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
) + (
    MathTool(
        operation_id="delta_matroid.dual.compute",
        title="Compute the dual of a finite delta-matroid",
        description="Return the complete feasible family obtained by complementing every feasible set on the retained ground axis.",
        request_type=DeltaMatroidDualRequest,
        result_type=FiniteDeltaMatroid,
        run=_run_dual,
        tags=("delta-matroid", "dual", "exact"),
        examples=(
            OperationExample(
                name="dual_uniform",
                description="Compute the dual by complementing feasible sets on the labelled ground.",
                input={
                    "delta_matroid": {
                        "ground": ["a", "b"],
                        "feasible": [[], [0], [0, 1], [1]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="delta_matroid.minor.compute",
        title="Compute a delta-matroid deletion/contraction minor",
        description="Return the exact feasible family after deleting and contracting disjoint ground-index sets.",
        request_type=DeltaMatroidMinorRequest,
        result_type=FiniteDeltaMatroid,
        run=_run_minor,
        tags=("delta-matroid", "minor", "exact"),
        examples=(
            OperationExample(
                name="delete_b",
                description="Delete ground element b; deletion indices must be disjoint from contractions.",
                input={
                    "delta_matroid": {
                        "ground": ["a", "b"],
                        "feasible": [[], [0], [0, 1], [1]],
                    },
                    "delete": [1],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="delta_matroid.from_binary_matrix.compute",
        title="Reconstruct a delta-matroid from binary principal minors",
        description=(
            "Compute every principal minor over GF(2) of a labelled symmetric binary matrix (at most 8 ground labels and 250,000 principal-elimination work units) and return the resulting feasible-set delta-matroid."
        ),
        request_type=BinaryMatrixRequest,
        result_type=BinaryMatrixResult,
        run=_run_binary,
        tags=("delta-matroid", "binary", "principal-minor", "exact"),
        examples=(
            OperationExample(
                name="binary_zero",
                description="Reconstruct the principal-minor delta-matroid of the zero 2-by-2 symmetric matrix.",
                input={"matrix": {"ground": ["a", "b"], "entries": [[0, 0], [0, 0]]}},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
