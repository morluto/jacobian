"""Finite delta-matroid operation declarations."""

from jacobian.catalog.models import (
    MathTool,
    MathTools,
    OperationDomainValidationError,
    OperationExample,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidDistanceRequest,
    DeltaMatroidDistanceResult,
    DeltaMatroidFromFeasibleSetsRequest,
    DeltaMatroidRecognitionResult,
    DeltaMatroidTwistRequest,
    DeltaMatroidWidthRequest,
    DeltaMatroidWidthResult,
)
from jacobian.math.combinatorics.matroids.delta.extra import (
    BinaryLoopComplementRequest,
    BinaryLoopComplementResult,
    BinaryMatrixRequest,
    BinaryMatrixResult,
    DeltaMatroidDirectSumRequest,
    DeltaMatroidDirectSumResult,
    DeltaMatroidDualRequest,
    DeltaMatroidFeasibleSizeProfile,
    DeltaMatroidFeasibleSizeProfileRequest,
    DeltaMatroidMinorRequest,
    DeltaMatroidTwistWidthProfileRequest,
    DeltaMatroidTwistWidthProfileResult,
)
from jacobian.math.combinatorics.matroids.delta.extra_ops import (
    binary,
    dual,
    feasible_size_profile,
    loop_complement,
    minor,
    twist_width_profile,
)
from jacobian.math.combinatorics.matroids.delta.operations import (
    direct_sum,
    distance,
    from_feasible_sets,
    twist,
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


def _run_loop_complement(
    request: BinaryLoopComplementRequest,
) -> BinaryLoopComplementResult:
    return loop_complement(request.matrix, request.subset)


def _run_direct_sum(
    request: DeltaMatroidDirectSumRequest,
) -> DeltaMatroidDirectSumResult:
    try:
        return direct_sum(request.left, request.right)
    except DeltaMatroidAdmissionError as exc:
        raise OperationResourceAdmissionError(
            location=("left",), code=f"delta_matroid.{exc.reason}", message=str(exc)
        ) from exc
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        raise
    except (TypeError, ValueError, IndexError) as exc:
        raise _extra_domain(("left",), "delta_matroid.source_not_valid", exc) from exc


def _run_twist_width_profile(
    request: DeltaMatroidTwistWidthProfileRequest,
) -> DeltaMatroidTwistWidthProfileResult:
    return twist_width_profile(request.delta_matroid)


def _run_feasible_size_profile(
    request: DeltaMatroidFeasibleSizeProfileRequest,
) -> DeltaMatroidFeasibleSizeProfile:
    return feasible_size_profile(request.delta_matroid)


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


def _distance(request: DeltaMatroidDistanceRequest) -> DeltaMatroidDistanceResult:
    try:
        return distance(request.delta_matroid, request.subset)
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
        operation_id="delta_matroid.direct_sum.compute",
        title="Take the direct sum of finite delta-matroids",
        description=(
            "Combine delta-matroids on disjoint labelled ground sets. Feasible "
            "sets are all pairwise unions, on the concatenated ground axis; the "
            "result includes both source index injections. Ground, pair-work, "
            "and membership cardinality bounds are checked first; symmetric "
            "exchange of the product follows from the direct-sum theorem."
        ),
        request_type=DeltaMatroidDirectSumRequest,
        result_type=DeltaMatroidDirectSumResult,
        run=_run_direct_sum,
        tags=("delta-matroid", "direct-sum", "exact"),
        examples=(
            OperationExample(
                name="direct_sum_singletons",
                description="Combine singleton feasible families on disjoint labelled elements.",
                input={
                    "left": {"ground": ["a"], "feasible": [[], [0]]},
                    "right": {"ground": ["b"], "feasible": [[], [0]]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="delta_matroid.distance.compute",
        title="Compute distance from a subset to delta-matroid feasibility",
        description=(
            "Return min |X symmetric_difference F| over the complete feasible "
            "family and the lexicographically first nearest feasible set."
        ),
        request_type=DeltaMatroidDistanceRequest,
        result_type=DeltaMatroidDistanceResult,
        run=_distance,
        tags=("delta-matroid", "distance", "feasibility", "exact"),
        examples=(
            OperationExample(
                name="distance_to_feasible",
                description="The empty set is distance one from the family {{a}}.",
                input={
                    "delta_matroid": {"ground": ["a"], "feasible": [[0]]},
                    "subset": [],
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
        operation_id="delta_matroid.twist_width_profile.compute",
        title="Compute the complete twist-width profile of a finite delta-matroid",
        description=(
            "Return the width after twisting by every ground subset, indexed by "
            "the subset's integer bit mask (bit i selects ground element i). "
            "Preflight admits at most 4,096 masks and 262,144 mask-feasible-set "
            "evaluations."
        ),
        request_type=DeltaMatroidTwistWidthProfileRequest,
        result_type=DeltaMatroidTwistWidthProfileResult,
        run=_run_twist_width_profile,
        tags=("delta-matroid", "twist", "width", "profile", "exact"),
        examples=(
            OperationExample(
                name="two_element_twist_widths",
                description="Return all four twist widths in mask order 0, 1, 2, 3.",
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
        operation_id="delta_matroid.feasible_size_profile.compute",
        title="Compute the feasible-set size profile of a finite delta-matroid",
        description=(
            "Return the exact ascending-size histogram (count at size 0, size 1, "
            "..., through the ground-set size), retaining the labelled ground "
            "axis. Profile output is admitted before source exchange validation."
        ),
        request_type=DeltaMatroidFeasibleSizeProfileRequest,
        result_type=DeltaMatroidFeasibleSizeProfile,
        run=_run_feasible_size_profile,
        tags=("delta-matroid", "feasible-sets", "size", "profile", "exact"),
        examples=(
            OperationExample(
                name="two_element_feasible_sizes",
                description=(
                    "The feasible family has one set of size zero, two of size "
                    "one, and one of size two."
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
        operation_id="delta_matroid.binary_loop_complement.compute",
        title="Loop complement a binary delta-matroid",
        description=(
            "Toggle the diagonal of a labelled symmetric GF(2) matrix on a "
            "sorted subset of its ground axis, then return the complete "
            "principal-minor feasible family of the resulting presentation. "
            "For one element this toggles feasibility of X union {e} for "
            "each currently feasible X not containing e. The same bounded "
            "principal-minor enumeration used by binary reconstruction applies."
        ),
        request_type=BinaryLoopComplementRequest,
        result_type=BinaryLoopComplementResult,
        run=_run_loop_complement,
        tags=("delta-matroid", "binary", "loop-complement", "exact"),
        examples=(
            OperationExample(
                name="loop_complement_zero_matrix",
                description="Toggle both diagonal entries of the zero matrix.",
                input={
                    "matrix": {"ground": ["a", "b"], "entries": [[0, 0], [0, 0]]},
                    "subset": [0, 1],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
