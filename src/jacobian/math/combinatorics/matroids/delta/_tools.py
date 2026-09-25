"""Finite delta-matroid operation declarations."""

from jacobian.catalog.models import (
    MathTool,
    MathTools,
    OperationDomainValidationError,
    OperationExample,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidDistanceProfileRequest,
    DeltaMatroidFromFeasibleSetsRequest,
    DeltaMatroidRecognitionResult,
    DeltaMatroidTwistRequest,
    DeltaMatroidWidthRequest,
    DeltaMatroidWidthResult,
)
from jacobian.math.combinatorics.matroids.delta.extra import (
    BinaryMatrixRequest,
    BinaryMatrixResult,
    BinaryMatrixTwistRequest,
    DeltaMatroidDualRequest,
    DeltaMatroidMinorRequest,
)
from jacobian.math.combinatorics.matroids.delta.extra_ops import (
    binary,
    binary_matrix_twist,
    dual,
    minor,
)
from jacobian.math.combinatorics.matroids.delta.operations import (
    distance_profile,
    from_feasible_sets,
    twist,
    width,
)
from jacobian.math.combinatorics.matroids.delta.values import (
    DeltaMatroidAdmissionError,
    DeltaMatroidDistanceProfile,
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


def _run_binary_twist(request: BinaryMatrixTwistRequest) -> BinaryMatrixResult:
    try:
        return binary_matrix_twist(request.matrix, request.subset)
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        raise
    except (TypeError, ValueError, IndexError) as exc:
        raise _extra_domain(
            ("matrix",), "delta_matroid.binary_twist_invalid", exc
        ) from exc


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


def _distance_profile(
    request: DeltaMatroidDistanceProfileRequest,
) -> DeltaMatroidDistanceProfile:
    try:
        return distance_profile(request.delta_matroid)
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


TOOLS: MathTools = (  # noqa: RUF005
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
        operation_id="delta_matroid.distance_profile.compute",
        title="Compute distance to feasibility for every ground subset",
        description=(
            "Return the exact Hamming distance from every subset of the "
            "ground set to its nearest feasible set, the number of nearest "
            "feasible sets, and the distance histogram. Subsets are ordered "
            "by integer mask with bit i denoting ground index i. Admission "
            "limits the complete profile to 4,096 masks and 262,144 "
            "subset/feasible-set evaluations, and replays source symmetric "
            "exchange under the existing 250,000-candidate bound."
        ),
        request_type=DeltaMatroidDistanceProfileRequest,
        result_type=DeltaMatroidDistanceProfile,
        run=_distance_profile,
        tags=("delta-matroid", "distance", "feasible-set-profile", "exact"),
        discovery_terms=("delta-matroid distance profile", "nearest feasible set"),
        examples=(
            OperationExample(
                name="distance_profile_of_two_element_cube",
                description=(
                    "The four feasible subsets give distance zero at each of "
                    "the four ground-subset masks."
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
    MathTool(
        operation_id="delta_matroid.binary.from_matrix_twist.compute",
        title="Construct a binary delta-matroid from a twisted matrix presentation",
        description=(
            "Return D(A)*T for a symmetric matrix A over GF(2) and a sorted "
            "ground-index subset T. It enumerates all nonsingular principal "
            "submatrices under the existing eight-element and 250,000-work "
            "bounds, then applies the exact symmetric-difference bijection. "
            "The source matrix, twist, and complete feasible family are retained; "
            "output has at most 256 rows and 1,024 memberships."
        ),
        request_type=BinaryMatrixTwistRequest,
        result_type=BinaryMatrixResult,
        run=_run_binary_twist,
        tags=("delta-matroid", "binary", "matrix-twist", "exact"),
        discovery_terms=(
            "binary delta-matroid twist",
            "twisted principal-minor family",
        ),
        examples=(
            OperationExample(
                name="twist_zero_matrix_by_both_elements",
                description=(
                    "The zero matrix has only the empty feasible set; twisting "
                    "by both axes gives the full two-element set."
                ),
                input={
                    "matrix": {
                        "ground": ["a", "b"],
                        "entries": [[0, 0], [0, 0]],
                    },
                    "subset": [0, 1],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
