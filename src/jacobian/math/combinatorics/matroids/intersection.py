from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids._models import (
    LinearMatroid,
    MatroidCommonBasisResult,
    MatroidIntersectionResult,
    MatroidIntersectionWitness,
    MatroidWeightedIntersectionCertificateRequest,
    MatroidWeightedIntersectionResult,
)
from jacobian.math.combinatorics.matroids.operations import (
    MAX_CLOSURE_RANK_WORK,
    _canonical_weight_function,
    _maximum_weight_independent_set_admitted,
    _prepare_maximum_weight_independent_set,
    _rank_work,
    _selected_columns_matrix,
)
from jacobian.math.matrices.finite_fields.linear_algebra import rank as pf_rank

MAX_INTERSECTION_GROUND = 256
# The oracle kernel has O(n^3) exchange probes.  A rank probe is charged for
# the larger representation row count; witness ranks and the result carrier
# are charged separately.  This is deliberately an admission bound, not a
# timeout: every admitted request has a finite exact completion envelope.
MAX_INTERSECTION_WORK = 50_000_000
MAX_INTERSECTION_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_WEIGHTED_INTERSECTION_WORK = 50_000_000
MAX_WEIGHTED_INTERSECTION_OUTPUT_BYTES = 8 * 1024 * 1024


def _admit_matroid(value: object, location: tuple[str, ...]) -> LinearMatroid:
    if type(value) is not LinearMatroid:
        raise OperationDomainValidationError(
            location=location,
            code="matroid.intersection.carrier",
            message="intersection operands must be canonical linear matroids",
        )
    try:
        # Revalidate model_construct-created values before indexing their
        # matrices.  Native calls do not receive Pydantic request validation.
        return LinearMatroid.model_validate(value.model_dump(mode="python"))
    except Exception as exc:
        raise OperationDomainValidationError(
            location=location,
            code="matroid.intersection.carrier",
            message="intersection operand is not a canonical linear matroid",
        ) from exc


def _admit_work(
    first: LinearMatroid,
    second: LinearMatroid,
    *,
    source_rank_calls: int = 0,
) -> None:
    n = first.ground_size
    rows = max(len(first.matrix.entries), len(second.matrix.entries), 1)
    # A rank of any queried subset costs at most rows*n*min(rows,n): the
    # selected matrix has at most n columns and rank's dense elimination
    # backend scales cubically in the smaller matrix axis. A breadth-first
    # search probes at most n outgoing edges from each of n vertices; there
    # are at most n augmentations and one final search. Charge every possible
    # probe at the larger operand's rank cost, even though the oracle cache
    # usually makes the actual count much smaller.
    rank_cost = rows * n * min(rows, n)
    exchange_rank_work = (n + 1) ** 3 * rank_cost
    # Four ranks produce the min-max witness and check both common-set
    # feasibility claims. Common-basis requests add both full-source ranks.
    final_rank_work = (4 + source_rank_calls) * rank_cost
    output_bytes = _intersection_output_bound_bytes(first, second)
    if (
        exchange_rank_work + final_rank_work > MAX_INTERSECTION_WORK
        or output_bytes > MAX_INTERSECTION_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("first", "second"),
            code="matroid.intersection.work_bound",
            message=(
                "intersection rank work or retained result output exceeds the "
                f"{MAX_INTERSECTION_WORK}-unit work or "
                f"{MAX_INTERSECTION_OUTPUT_BYTES}-byte output envelope"
            ),
        )


def _intersection_output_bound_bytes(
    first: LinearMatroid, second: LinearMatroid
) -> int:
    """Conservatively bound the JSON size of source-bound intersection output."""

    def source_bytes(matroid: LinearMatroid) -> int:
        rows = len(matroid.matrix.entries)
        columns = matroid.ground_size
        # Residues have at most ten decimal digits. The extra per cell covers
        # separators; row/field/container syntax is covered by the fixed slack.
        matrix_bytes = rows * columns * 11 + rows * 4 + 256
        labels = matroid.ground_axis
        # JSON may escape controls as six characters and non-BMP scalars as
        # UTF-16 surrogate pairs. Charge the larger twelve-byte bound per
        # Python code point, plus quotes and separators.
        axis_bytes = sum(12 * len(label) + 3 for label in labels)
        return matrix_bytes + axis_bytes

    n = first.ground_size
    # Both retained source matrices/axes, two O(n) index tuples and scalar
    # fields. Indexes need at most three decimal digits at the admitted axis.
    return source_bytes(first) + source_bytes(second) + 24 * n + 1024


def _independent(m: LinearMatroid, subset: Sequence[int]) -> bool:
    return len(subset) == pf_rank(_selected_columns_matrix(m, list(subset)))


def _rank(m: LinearMatroid, indices: Sequence[int]) -> int:
    return pf_rank(_selected_columns_matrix(m, list(indices))) if indices else 0


def _weighted_intersection_output_bound_bytes(
    first: LinearMatroid,
    second: LinearMatroid,
    objective: Sequence[int],
    first_split: Sequence[int],
    second_split: Sequence[int],
    candidate_size: int,
) -> int:
    """Conservative JSON-size bound for the retained source-bound result."""

    def axis_bound(matroid: LinearMatroid) -> int:
        # json escapes at most six ASCII characters per control character and
        # twelve for a non-BMP scalar represented as a UTF-16 surrogate pair.
        return sum(12 * len(label) + 3 for label in matroid.ground_axis)

    def source_bound(matroid: LinearMatroid) -> int:
        rows = len(matroid.matrix.entries)
        columns = matroid.ground_size
        # Each residue has at most 10 decimal digits; 16 also covers commas.
        return 16 * rows * columns + 4 * rows + 256 + axis_bound(matroid)

    def weights_bound(matroid: LinearMatroid, values: Sequence[int]) -> int:
        return (
            axis_bound(matroid)
            + sum(len(str(abs(value))) + 1 for value in values)
            + 128
        )

    def maximum_result_bound(matroid: LinearMatroid, values: Sequence[int]) -> int:
        n = matroid.ground_size
        return source_bound(matroid) + weights_bound(matroid, values) + 10 * n + 256

    return (
        maximum_result_bound(first, first_split)
        + maximum_result_bound(second, second_split)
        + weights_bound(first, objective)
        + 4 * candidate_size
        + 512
    )


def replay_intersection_result(result: MatroidIntersectionResult) -> None:
    """Replay a caller-supplied intersection claim against its retained sources.

    The returned common set is a feasible lower bound only after both source
    ranks equal its cardinality. The retained partition ranks must also match
    their source restrictions; equality with the lower bound then proves the
    min-max optimum independently of any search trajectory.
    """

    first = _admit_matroid(result.first, ("first",))
    second = _admit_matroid(result.second, ("second",))
    if (
        first.matrix.prime != second.matrix.prime
        or first.ground_axis != second.ground_axis
        or first.ground_size != second.ground_size
    ):
        raise ValueError("intersection sources must share one ground and field")
    _admit_work(first, second)
    common = result.common_independent
    if (
        _rank(first, common) != result.rank_first_common
        or _rank(second, common) != result.rank_second_common
    ):
        raise ValueError("common-set rank claims disagree with their source matroids")
    complement = tuple(
        i for i in range(first.ground_size) if i not in set(result.witness.subset)
    )
    if (
        _rank(first, result.witness.subset) != result.witness.rank_first
        or _rank(second, complement) != result.witness.rank_second_complement
    ):
        raise ValueError("min-max rank claims disagree with their source matroids")


def _matroid_intersection_admitted(  # noqa: C901
    first: LinearMatroid, second: LinearMatroid
) -> MatroidIntersectionResult:
    """Exact augmenting-path kernel; caller owns source and work admission."""
    n = first.ground_size
    # Edmonds' augmenting-path algorithm with linear-matroid independence
    # oracles.  Caching makes the charged O(n^3) oracle envelope meaningful
    # even when several exchange paths inspect the same subset.
    chosen_set: set[int] = set()
    cache: dict[tuple[int, tuple[int, ...]], bool] = {}

    def independent(which: int, subset: Sequence[int]) -> bool:
        key = (which, tuple(sorted(subset)))
        if key not in cache:
            cache[key] = _independent(first if which == 1 else second, key[1])
        return cache[key]

    def search() -> tuple[list[int], set[int]]:
        source = -1
        sink = n
        parent: dict[int, int | None] = {source: None}
        queue: deque[int] = deque([source])
        while queue and sink not in parent:
            node = queue.popleft()
            if node == source:
                next_nodes = [
                    x
                    for x in range(n)
                    if x not in chosen_set and independent(1, (*chosen_set, x))
                ]
            elif node not in chosen_set:
                next_nodes = [
                    y
                    for y in sorted(chosen_set)
                    if independent(2, sorted((chosen_set | {node}) - {y}))
                ]
                if independent(2, (*chosen_set, node)):
                    next_nodes.append(sink)
            else:
                next_nodes = [
                    x
                    for x in range(n)
                    if x not in chosen_set
                    and independent(1, sorted((chosen_set | {x}) - {node}))
                ]
            for target in next_nodes:
                if target not in parent:
                    parent[target] = node
                    queue.append(target)
        if sink not in parent:
            return [], set(parent) - {source, sink}
        path: list[int] = []
        node = sink
        while parent[node] != source:
            node = parent[node]  # type: ignore[assignment]
            path.append(node)
        path.reverse()
        return path, set(parent) - {source, sink}

    while True:
        path, _ = search()
        if not path:
            break
        chosen_set.symmetric_difference_update(path)

    _, reachable = search()
    independent_set = tuple(sorted(chosen_set))
    subset = tuple(i for i in range(n) if i not in reachable)
    rank1 = _rank(first, subset)
    rank2 = _rank(second, tuple(i for i in range(n) if i not in subset))
    if rank1 + rank2 != len(independent_set):
        raise OperationDomainValidationError(
            location=("first", "second"),
            code="matroid.intersection.witness",
            message="exchange kernel did not establish its min-max witness",
        )
    rank1_common = _rank(first, independent_set)
    rank2_common = _rank(second, independent_set)
    if rank1_common != len(independent_set) or rank2_common != len(independent_set):
        raise OperationDomainValidationError(
            location=("first", "second", "common_independent"),
            code="matroid.intersection.feasibility",
            message="exchange kernel did not return a common independent set",
        )
    witness = MatroidIntersectionWitness(
        subset=subset,
        rank_first=rank1,
        rank_second_complement=rank2,
        equality=rank1 + rank2,
    )
    return MatroidIntersectionResult._from_kernel(
        first=first,
        second=second,
        common_independent=independent_set,
        cardinality=len(independent_set),
        rank_first_common=rank1_common,
        rank_second_common=rank2_common,
        witness=witness,
    )


def _admit_pair(first: object, second: object) -> tuple[LinearMatroid, LinearMatroid]:
    first = _admit_matroid(first, ("first",))
    second = _admit_matroid(second, ("second",))
    if (
        first.matrix.prime != second.matrix.prime
        or first.ground_axis != second.ground_axis
        or first.ground_size != second.ground_size
    ):
        raise OperationDomainValidationError(
            location=("second",),
            code="matroid.intersection.foreign_ground",
            message="matroids must share one labelled ground axis and field",
        )
    if first.ground_size > MAX_INTERSECTION_GROUND:
        raise OperationResourceAdmissionError(
            location=("first", "matrix"),
            code="matroid.intersection.work_bound",
            message=f"intersection ground is limited to {MAX_INTERSECTION_GROUND} elements",
        )
    return first, second


def matroid_intersection(
    first: LinearMatroid, second: LinearMatroid
) -> MatroidIntersectionResult:
    """Compute one maximum common independent set and Edmonds witness."""
    first, second = _admit_pair(first, second)
    _admit_work(first, second)
    return _matroid_intersection_admitted(first, second)


def matroid_common_basis(
    first: LinearMatroid, second: LinearMatroid
) -> MatroidCommonBasisResult:
    """Return a closed common-basis decision with the exact max-intersection proof."""
    first, second = _admit_pair(first, second)
    # The one operation admission includes exchange, witness, feasibility, and
    # the two added source-rank computations before any exact expansion.
    _admit_work(first, second, source_rank_calls=2)
    maximum = _matroid_intersection_admitted(first, second)
    return MatroidCommonBasisResult._from_kernel(
        intersection=maximum,
        rank_first=_rank(first, tuple(range(first.ground_size))),
        rank_second=_rank(second, tuple(range(second.ground_size))),
    )


def replay_common_basis_result(result: MatroidCommonBasisResult) -> None:
    """Check a serialized common-basis outcome and all retained source ranks."""
    first, second = _admit_pair(result.first, result.second)
    _admit_work(first, second, source_rank_calls=2)
    common = result.common_independent
    witness_subset = result.witness.subset
    if (
        common != tuple(sorted(set(common)))
        or any(not 0 <= i < first.ground_size for i in common)
        or result.cardinality != len(common)
        or result.rank_first_common != result.cardinality
        or result.rank_second_common != result.cardinality
        or witness_subset != tuple(sorted(set(witness_subset)))
        or any(not 0 <= i < first.ground_size for i in witness_subset)
    ):
        raise ValueError("common-basis indices or cardinality are not canonical")
    complement = tuple(
        i for i in range(first.ground_size) if i not in set(witness_subset)
    )
    replayed = (
        _rank(first, common),
        _rank(second, common),
        _rank(first, witness_subset),
        _rank(second, complement),
        _rank(first, tuple(range(first.ground_size))),
        _rank(second, tuple(range(second.ground_size))),
    )
    claimed = (
        result.rank_first_common,
        result.rank_second_common,
        result.witness.rank_first,
        result.witness.rank_second_complement,
        result.rank_first,
        result.rank_second,
    )
    if replayed != claimed:
        raise ValueError("common-basis rank claims disagree with source matrices")
    if (
        result.witness.equality != result.cardinality
        or result.witness.rank_first + result.witness.rank_second_complement
        != result.cardinality
    ):
        raise ValueError("common-basis min-max witness does not prove the maximum")
    expected_status = (
        "COMMON_BASIS"
        if result.rank_first == result.rank_second == result.cardinality
        else "NO_COMMON_BASIS"
    )
    expected_reason = (
        "COMMON_BASIS"
        if expected_status == "COMMON_BASIS"
        else "SOURCE_RANK_MISMATCH"
        if result.rank_first != result.rank_second
        else "MAXIMUM_COMMON_INDEPENDENT_SET_TOO_SMALL"
    )
    expected_basis = (
        result.common_independent if expected_status == "COMMON_BASIS" else None
    )
    if (
        result.status != expected_status
        or result.reason != expected_reason
        or result.common_basis != expected_basis
    ):
        raise ValueError("common-basis status does not follow exact source ranks")


def verify_common_basis_result(result: MatroidCommonBasisResult) -> bool:
    """Return whether the retained values replay to the claimed common-basis outcome."""
    try:
        replay_common_basis_result(result)
        return True
    except (
        OperationDomainValidationError,
        OperationResourceAdmissionError,
        TypeError,
        ValueError,
    ):
        return False


def weighted_intersection_certificate(
    request: MatroidWeightedIntersectionCertificateRequest,
) -> MatroidWeightedIntersectionResult:
    """Check a supplied integral weight-splitting certificate for a candidate.

    This operation does not search for a common independent set or a weight
    split. It recomputes both source matroids' exact split-weight maxima by the
    bounded greedy kernel and accepts only when their sum equals the feasible
    candidate's objective value.
    """
    if type(request) is not MatroidWeightedIntersectionCertificateRequest:
        raise OperationDomainValidationError(
            location=("request",),
            code="matroid.weighted_intersection.request",
            message="request must be a canonical weighted-intersection certificate",
        )
    try:
        request = MatroidWeightedIntersectionCertificateRequest.model_validate(
            request.model_dump(mode="python")
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="matroid.weighted_intersection.request",
            message="weighted-intersection certificate request is not canonical",
        ) from exc

    first, second = _admit_pair(request.first, request.second)
    objective, _ = _canonical_weight_function(first, request.weight_function)
    first_split, first_split_function = _canonical_weight_function(
        first, request.first_split
    )
    second_split, second_split_function = _canonical_weight_function(
        second, request.second_split
    )
    if (
        tuple(a + b for a, b in zip(first_split, second_split, strict=True))
        != objective
    ):
        raise OperationDomainValidationError(
            location=("first_split", "second_split"),
            code="matroid.weighted_intersection.split",
            message="integral dual weights must sum coordinatewise to the objective",
        )

    candidate = request.common_independent
    candidate_size = len(candidate)
    candidate_weight = sum(objective[index] for index in candidate)
    first_rank_work = _rank_work(len(first.matrix.entries), candidate_size)
    second_rank_work = _rank_work(len(second.matrix.entries), candidate_size)
    first_values, first_function, first_work, first_output = (
        _prepare_maximum_weight_independent_set(first, first_split_function)
    )
    second_values, second_function, second_work, second_output = (
        _prepare_maximum_weight_independent_set(second, second_split_function)
    )
    total_work = first_work + second_work + first_rank_work + second_rank_work
    output_bytes = _weighted_intersection_output_bound_bytes(
        first,
        second,
        objective,
        first_split,
        second_split,
        candidate_size,
    )
    if (
        first_work > MAX_CLOSURE_RANK_WORK
        or second_work > MAX_CLOSURE_RANK_WORK
        or first_output > 16 * 256
        or second_output > 16 * 256
        or total_work > MAX_WEIGHTED_INTERSECTION_WORK
        or output_bytes > MAX_WEIGHTED_INTERSECTION_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("first", "second", "weights"),
            code="matroid.weighted_intersection.work_bound",
            message=(
                "weighted-intersection certificate rank work or result output "
                f"exceeds the {MAX_WEIGHTED_INTERSECTION_WORK}-unit work or "
                f"{MAX_WEIGHTED_INTERSECTION_OUTPUT_BYTES}-byte output envelope"
            ),
        )
    # Both split phases and feasibility probes are admitted together before
    # the first exact rank expansion.
    first_candidate_rank = _rank(first, candidate)
    second_candidate_rank = _rank(second, candidate)
    if (
        first_candidate_rank != candidate_size
        or second_candidate_rank != candidate_size
    ):
        raise OperationDomainValidationError(
            location=("common_independent",),
            code="matroid.weighted_intersection.feasibility",
            message="candidate must be independent in both source matroids",
        )

    first_maximizer = _maximum_weight_independent_set_admitted(
        first, first_values, first_function
    )
    second_maximizer = _maximum_weight_independent_set_admitted(
        second, second_values, second_function
    )
    if first_maximizer.total_weight + second_maximizer.total_weight != candidate_weight:
        raise OperationDomainValidationError(
            location=("common_independent", "first_split", "second_split"),
            code="matroid.weighted_intersection.optimality",
            message=(
                "the supplied integral split does not certify the candidate's "
                "maximum common-independent-set weight"
            ),
        )
    _, canonical_objective = _canonical_weight_function(first, request.weight_function)
    return MatroidWeightedIntersectionResult._from_kernel(
        weight_function=canonical_objective,
        common_independent=candidate,
        total_weight=candidate_weight,
        first_maximizer=first_maximizer,
        second_maximizer=second_maximizer,
    )


def verify_weighted_intersection_result(
    result: MatroidWeightedIntersectionResult,
) -> bool:
    """Explicitly recompute a serialized weighted-intersection certificate."""
    try:
        request = MatroidWeightedIntersectionCertificateRequest(
            first=result.first,
            second=result.second,
            weight_function=result.weight_function,
            common_independent=result.common_independent,
            first_split=result.first_maximizer.weight_function,
            second_split=result.second_maximizer.weight_function,
        )
        return weighted_intersection_certificate(request) == result
    except (
        OperationDomainValidationError,
        OperationResourceAdmissionError,
        TypeError,
        ValueError,
    ):
        return False


__all__ = [
    "matroid_common_basis",
    "matroid_intersection",
    "replay_common_basis_result",
    "replay_intersection_result",
    "verify_common_basis_result",
    "verify_weighted_intersection_result",
    "weighted_intersection_certificate",
]
