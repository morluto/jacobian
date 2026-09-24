from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from math import ceil, log10

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids._models import (
    MAX_WEIGHTED_INTERSECTION_OPT_DUAL_DIGITS,
    LinearMatroid,
    MatroidCommonBasisResult,
    MatroidIntersectionResult,
    MatroidIntersectionWitness,
    MatroidRankMultiplier,
    MatroidWeightedIntersectionCertificateRequest,
    MatroidWeightedIntersectionOptimizationRequest,
    MatroidWeightedIntersectionOptimizationResult,
    MatroidWeightedIntersectionRankCertificateRequest,
    MatroidWeightedIntersectionRankCertificateResult,
    MatroidWeightedIntersectionResult,
    MatroidWeightFunction,
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


def _weighted_intersection_optimization_admission(
    first: LinearMatroid,
    second: LinearMatroid,
    objective: Sequence[int],
) -> None:
    """Admit Frank's integer weight-splitting kernel before rank expansion.

    For n elements, the exchange circuits are constructed once at each of at
    most n+1 cardinality stages. Schrijver's termination proof bounds the
    number of dual adjustments between augmentations by n, so at most
    n(n+1) graph/slack rounds occur. A round scans O(n²) arcs. Integral input
    weights keep all split values integral; each finite adjustment slack is a
    positive integer.
    """
    n = first.ground_size
    if n > MAX_INTERSECTION_GROUND:
        raise OperationResourceAdmissionError(
            location=("first", "matrix"),
            code="matroid.weighted_intersection.optimize.work_bound",
            message=(
                "weighted intersection ground is limited to "
                f"{MAX_INTERSECTION_GROUND} elements"
            ),
        )

    rows_first = max(1, len(first.matrix.entries))
    rows_second = max(1, len(second.matrix.entries))
    rank_cost_first = _rank_work(rows_first, n)
    rank_cost_second = _rank_work(rows_second, n)
    # At each cardinality stage, each outside element needs one X+y rank test
    # and at most |X| removal tests for each matroid. Include final feasibility
    # ranks; all requests are admitted before the first such test.
    calls_per_source = n * (n + 1) ** 2 + 1
    rank_work = calls_per_source * (rank_cost_first + rank_cost_second)

    update_rounds = n * (n + 1)
    scan_visits = 16 * n * n * update_rounds + 4 * n * n * (n + 1) + 2 * n * n
    weight_digits = max(
        (len(str(abs(value))) for value in objective),
        default=1,
    )
    # If M=max(1, |w_e|), each epsilon is at most 2(M+max|c1|).
    # Thus max|c1| evolves by B' <= 3B+2M. With at most n(n+1)
    # adjustments, both split vectors remain below 2*3^T*M.
    dual_digits = weight_digits + ceil(log10(2) + update_rounds * log10(3))
    if dual_digits > MAX_WEIGHTED_INTERSECTION_OPT_DUAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("weight_function", "values"),
            code="matroid.weighted_intersection.optimize.growth_bound",
            message=(
                "Frank weight-split intermediates exceed the "
                f"{MAX_WEIGHTED_INTERSECTION_OPT_DUAL_DIGITS}-digit bound"
            ),
        )
    # A decimal limb holds at most nine digits. Charge every graph/slack scan
    # at the maximum possible integer width; this bounds Python big-int
    # comparisons, additions, and subtractions independently of wall time.
    arithmetic_work = scan_visits * max(1, ceil(dual_digits / 9))

    output_bytes = (
        _intersection_output_bound_bytes(first, second)
        + sum(12 * len(label) + 3 for label in first.ground_axis)
        + sum(len(str(abs(value))) + 3 for value in objective)
        + 8 * n
        + 512
    )
    if (
        rank_work + arithmetic_work > MAX_WEIGHTED_INTERSECTION_WORK
        or output_bytes > MAX_WEIGHTED_INTERSECTION_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("first", "second", "weight_function"),
            code="matroid.weighted_intersection.optimize.work_bound",
            message=(
                "weighted intersection rank work, exact integer arithmetic, "
                f"or result output exceeds the {MAX_WEIGHTED_INTERSECTION_WORK}-unit "
                f"work or {MAX_WEIGHTED_INTERSECTION_OUTPUT_BYTES}-byte output envelope"
            ),
        )


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


def _weighted_matroid_intersection_admitted(
    first: LinearMatroid,
    second: LinearMatroid,
    weights: tuple[int, ...],
) -> tuple[int, ...]:
    """Run Frank's exact weight-splitting augmenting algorithm.

    `c1+c2=weights` is maintained throughout. At each cardinality, the
    current common independent set is optimal for each split weight among
    sets of that size. Tight exchange paths augment that cardinality profile;
    when no tight path exists, the minimum positive integral slack adjusts
    the split on the reachable side. An infinite slack proves there is no
    larger common independent set. The caller admits every circuit rank,
    graph scan, split intermediate, and the retained result before entry.
    """
    n = first.ground_size
    chosen: set[int] = set()
    first_split = list(weights)
    second_split = [0] * n
    profile: list[tuple[tuple[int, ...], int]] = [((), 0)]
    updates = 0
    augmentations = 0

    while True:
        chosen_tuple = tuple(sorted(chosen))
        first_arcs, second_arcs, sources, sinks = _weighted_exchange_graph(
            first, second, chosen_tuple, n
        )

        # No element can be added to one of the matroids, so the current set
        # already has maximum possible common cardinality. Its profile through
        # the current size includes every feasible common cardinality.
        if not sources or not sinks:
            break

        # The exchange graph and its arc families depend only on the current
        # common set. Dual adjustments revisit this data without replaying
        # circuit ranks.
        while True:
            max_first_source = max(first_split[element] for element in sources)
            max_second_sink = max(second_split[element] for element in sinks)
            normalized_sources = tuple(
                element
                for element in sources
                if first_split[element] == max_first_source
            )
            normalized_sinks = {
                element for element in sinks if second_split[element] == max_second_sink
            }

            adjacency = _weighted_tight_adjacency(
                first_arcs, second_arcs, first_split, second_split, n
            )
            path, reachable = _weighted_tight_path(
                adjacency, normalized_sources, normalized_sinks
            )

            if path is not None:
                chosen.symmetric_difference_update(path)
                augmentations += 1
                if augmentations > n:
                    raise OperationDomainValidationError(
                        location=("first", "second"),
                        code="matroid.weighted_intersection.optimizer_invariant",
                        message="weighted intersection exceeded the ground-size augmentation bound",
                    )
                selected = tuple(sorted(chosen))
                profile.append((selected, sum(weights[index] for index in selected)))
                break

            slacks = _weighted_exchange_slacks(
                first_arcs,
                second_arcs,
                sources,
                sinks,
                reachable,
                max_first_source,
                max_second_sink,
                first_split,
                second_split,
            )

            if not slacks:
                # All four Frank slack families are empty. The full exchange
                # graph has no source-to-sink path, so this is a maximum
                # cardinality common set and the computed profile is complete.
                return min(
                    profile,
                    key=lambda item: (-item[1], len(item[0]), item[0]),
                )[0]
            epsilon = min(slacks)
            if epsilon <= 0:
                raise OperationDomainValidationError(
                    location=("first", "second"),
                    code="matroid.weighted_intersection.optimizer_invariant",
                    message="weighted exchange slacks must be strictly positive",
                )
            for element in reachable:
                first_split[element] -= epsilon
                second_split[element] += epsilon
            updates += 1
            if updates > n * (n + 1):
                raise OperationDomainValidationError(
                    location=("first", "second"),
                    code="matroid.weighted_intersection.optimizer_invariant",
                    message="weighted intersection exceeded Frank's dual-update bound",
                )

    return min(profile, key=lambda item: (-item[1], len(item[0]), item[0]))[0]


def _weighted_exchange_graph(
    first: LinearMatroid,
    second: LinearMatroid,
    chosen: tuple[int, ...],
    ground_size: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int], list[int]]:
    first_arcs: list[tuple[int, int]] = []
    second_arcs: list[tuple[int, int]] = []
    sources: list[int] = []
    sinks: list[int] = []
    for outside in range(ground_size):
        if outside in chosen:
            continue
        plus = (*chosen, outside)
        first_addable = _independent(first, plus)
        second_addable = _independent(second, plus)
        if first_addable:
            sources.append(outside)
        if second_addable:
            sinks.append(outside)
        if first_addable and second_addable:
            continue
        for index, inside in enumerate(chosen):
            exchanged = (*chosen[:index], *chosen[index + 1 :], outside)
            if not first_addable and _independent(first, exchanged):
                first_arcs.append((inside, outside))
            if not second_addable and _independent(second, exchanged):
                second_arcs.append((outside, inside))
    return first_arcs, second_arcs, sources, sinks


def _weighted_tight_adjacency(
    first_arcs: list[tuple[int, int]],
    second_arcs: list[tuple[int, int]],
    first_split: list[int],
    second_split: list[int],
    ground_size: int,
) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(ground_size)]
    for inside, outside in first_arcs:
        if first_split[inside] == first_split[outside]:
            adjacency[inside].append(outside)
    for outside, inside in second_arcs:
        if second_split[inside] == second_split[outside]:
            adjacency[outside].append(inside)
    return adjacency


def _weighted_tight_path(
    adjacency: list[list[int]],
    sources: tuple[int, ...],
    sinks: set[int],
) -> tuple[tuple[int, ...] | None, set[int]]:
    parent: dict[int, int | None] = dict.fromkeys(sources)
    queue: deque[int] = deque(sources)
    endpoint: int | None = None
    while queue and endpoint is None:
        node = queue.popleft()
        if node in sinks:
            endpoint = node
            break
        for target in adjacency[node]:
            if target not in parent:
                parent[target] = node
                queue.append(target)
    if endpoint is None:
        return None, set(parent)
    path: list[int] = []
    path_node: int | None = endpoint
    while path_node is not None:
        path.append(path_node)
        path_node = parent[path_node]
    return tuple(path), set(parent)


def _weighted_exchange_slacks(
    first_arcs: list[tuple[int, int]],
    second_arcs: list[tuple[int, int]],
    sources: list[int],
    sinks: list[int],
    reachable: set[int],
    max_first_source: int,
    max_second_sink: int,
    first_split: list[int],
    second_split: list[int],
) -> list[int]:
    return [
        *(
            first_split[inside] - first_split[outside]
            for inside, outside in first_arcs
            if inside in reachable and outside not in reachable
        ),
        *(
            second_split[inside] - second_split[outside]
            for outside, inside in second_arcs
            if outside in reachable and inside not in reachable
        ),
        *(
            max_first_source - first_split[element]
            for element in sources
            if element not in reachable
        ),
        *(
            max_second_sink - second_split[element]
            for element in sinks
            if element in reachable
        ),
    ]


def maximum_weight_matroid_intersection(
    request: MatroidWeightedIntersectionOptimizationRequest,
) -> MatroidWeightedIntersectionOptimizationResult:
    """Compute one maximum-weight common independent set exactly.

    The solver uses Frank's integral weight-splitting augmenting algorithm.
    The independently published supplied-certificate operations remain
    available to check authored split or rank-dual witnesses.
    """
    if type(request) is not MatroidWeightedIntersectionOptimizationRequest:
        raise OperationDomainValidationError(
            location=("request",),
            code="matroid.weighted_intersection.optimize.request",
            message="request must be a canonical weighted-intersection optimization request",
        )
    try:
        request = MatroidWeightedIntersectionOptimizationRequest.model_validate(
            request.model_dump(mode="python")
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="matroid.weighted_intersection.optimize.request",
            message="weighted-intersection optimization request is not canonical",
        ) from exc

    first, second = _admit_pair(request.first, request.second)
    weights, canonical_function = _canonical_weight_function(
        first, request.weight_function
    )
    _weighted_intersection_optimization_admission(first, second, weights)
    selected = _weighted_matroid_intersection_admitted(first, second, weights)
    rank_first = _rank(first, selected)
    rank_second = _rank(second, selected)
    if rank_first != len(selected) or rank_second != len(selected):
        raise OperationDomainValidationError(
            location=("common_independent",),
            code="matroid.weighted_intersection.optimizer_invariant",
            message="weighted-intersection kernel did not return a common independent set",
        )
    total_weight = sum(weights[index] for index in selected)
    canonical_request = MatroidWeightedIntersectionOptimizationRequest.model_construct(
        first=first,
        second=second,
        weight_function=canonical_function,
    )
    return MatroidWeightedIntersectionOptimizationResult._from_kernel(
        request=canonical_request,
        common_independent=selected,
        total_weight=total_weight,
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


def _weighted_rank_dual_ranks(
    sources: tuple[LinearMatroid, LinearMatroid],
    families: tuple[
        tuple[MatroidRankMultiplier, ...], tuple[MatroidRankMultiplier, ...]
    ],
    ground_size: int,
    maximum_positive_weight: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Recompute listed ranks and enforce their coefficient bounds."""
    ranks_by_side: list[tuple[int, ...]] = []
    for source, family in zip(sources, families, strict=True):
        ranks = tuple(_rank(source, term.subset) for term in family)
        for term, rank in zip(family, ranks, strict=True):
            bound = maximum_positive_weight
            if rank > 0:
                bound *= ground_size
            if term.multiplier > bound:
                raise OperationDomainValidationError(
                    location=("rank_terms",),
                    code="matroid.weighted_intersection.rank_dual.multiplier",
                    message=(
                        "rank multiplier exceeds the normalized coefficient "
                        "bound for its rank and objective"
                    ),
                )
        ranks_by_side.append(ranks)
    return ranks_by_side[0], ranks_by_side[1]


def _weighted_rank_dual_values_and_cover(
    families: tuple[
        tuple[MatroidRankMultiplier, ...], tuple[MatroidRankMultiplier, ...]
    ],
    ranks_by_side: tuple[tuple[int, ...], tuple[int, ...]],
    ground_size: int,
) -> tuple[int, list[int]]:
    """Return the rank-dual objective and its coordinatewise coverage."""
    dual_value = 0
    coverage = [0] * ground_size
    for family, ranks in zip(families, ranks_by_side, strict=True):
        dual_value += sum(
            term.multiplier * rank for term, rank in zip(family, ranks, strict=True)
        )
        for term in family:
            for element in term.subset:
                coverage[element] += term.multiplier
    return dual_value, coverage


def weighted_intersection_rank_certificate(
    request: MatroidWeightedIntersectionRankCertificateRequest,
) -> MatroidWeightedIntersectionRankCertificateResult:
    """Check a sparse rank-inequality dual for a common independent set.

    The dual bound is computed from the caller's listed rank multipliers. No
    independent-set enumeration or optimizer is used. The returned split is
    derived from the first-side dual coverage, clipped coordinatewise to the
    positive objective weights, and can be passed to the existing supplied
    weight-splitting checker.
    """
    if type(request) is not MatroidWeightedIntersectionRankCertificateRequest:
        raise OperationDomainValidationError(
            location=("request",),
            code="matroid.weighted_intersection.rank_dual.request",
            message="request must be a canonical weighted-intersection rank certificate",
        )
    try:
        request = MatroidWeightedIntersectionRankCertificateRequest.model_validate(
            request.model_dump(mode="python")
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="matroid.weighted_intersection.rank_dual.request",
            message="weighted-intersection rank certificate request is not canonical",
        ) from exc

    first, second = _admit_pair(request.first, request.second)
    objective, _ = _canonical_weight_function(first, request.weight_function)
    n = first.ground_size
    w_plus = max(0, max(objective, default=0))
    candidate = request.common_independent
    terms = (request.first_rank_terms, request.second_rank_terms)
    sources = (first, second)
    rank_costs = tuple(
        sum(_rank_work(len(source.matrix.entries), len(term.subset)) for term in family)
        + _rank_work(len(source.matrix.entries), len(candidate))
        for source, family in zip(sources, terms, strict=True)
    )
    rank_work = sum(rank_costs)
    cover_work = n * sum(len(family) for family in terms)
    split_values_first = [0] * n
    for term in request.first_rank_terms:
        for element in term.subset:
            split_values_first[element] += term.multiplier
    split_values_first = [
        min(value, max(0, objective[element]))
        for element, value in enumerate(split_values_first)
    ]
    split_values_second = [
        objective[element] - split_values_first[element] for element in range(n)
    ]
    first_split = MatroidWeightFunction(
        ground_axis=first.ground_axis,
        values=tuple(split_values_first),
    )
    second_split = MatroidWeightFunction(
        ground_axis=second.ground_axis,
        values=tuple(split_values_second),
    )
    output_bytes = (
        _weighted_intersection_output_bound_bytes(
            first,
            second,
            objective,
            split_values_first,
            split_values_second,
            len(candidate),
        )
        + 8 * sum(len(term.subset) for family in terms for term in family)
        + 64 * sum(len(family) for family in terms)
    )
    total_work = rank_work + cover_work
    if (
        rank_work > MAX_WEIGHTED_INTERSECTION_WORK
        or total_work > MAX_WEIGHTED_INTERSECTION_WORK
        or output_bytes > MAX_WEIGHTED_INTERSECTION_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("first", "second", "rank_terms"),
            code="matroid.weighted_intersection.rank_dual.work_bound",
            message=(
                "rank-dual certificate rank work or result output exceeds the "
                f"{MAX_WEIGHTED_INTERSECTION_WORK}-unit work or "
                f"{MAX_WEIGHTED_INTERSECTION_OUTPUT_BYTES}-byte output envelope"
            ),
        )

    ranks_by_side = _weighted_rank_dual_ranks(sources, terms, n, w_plus)

    first_candidate_rank = _rank(first, candidate)
    second_candidate_rank = _rank(second, candidate)
    if first_candidate_rank != len(candidate) or second_candidate_rank != len(
        candidate
    ):
        raise OperationDomainValidationError(
            location=("common_independent",),
            code="matroid.weighted_intersection.rank_dual.feasibility",
            message="candidate must be independent in both source matroids",
        )

    dual_value, coverage = _weighted_rank_dual_values_and_cover(terms, ranks_by_side, n)
    if any(coverage[element] < objective[element] for element in range(n)):
        raise OperationDomainValidationError(
            location=("first_rank_terms", "second_rank_terms"),
            code="matroid.weighted_intersection.rank_dual.cover",
            message="rank-dual multipliers must cover every objective weight coordinatewise",
        )

    candidate_weight = sum(objective[element] for element in candidate)
    if dual_value != candidate_weight:
        raise OperationDomainValidationError(
            location=("common_independent", "first_rank_terms", "second_rank_terms"),
            code="matroid.weighted_intersection.rank_dual.optimality",
            message="rank-dual objective must equal the feasible candidate weight",
        )
    return MatroidWeightedIntersectionRankCertificateResult._from_kernel(
        request=request,
        total_weight=candidate_weight,
        first_split=first_split,
        second_split=second_split,
    )


def verify_weighted_intersection_rank_certificate(
    result: MatroidWeightedIntersectionRankCertificateResult,
) -> bool:
    """Explicitly recompute a serialized rank-dual certificate."""
    try:
        request = MatroidWeightedIntersectionRankCertificateRequest(
            first=result.first,
            second=result.second,
            weight_function=result.weight_function,
            common_independent=result.common_independent,
            first_rank_terms=result.first_rank_terms,
            second_rank_terms=result.second_rank_terms,
        )
        return weighted_intersection_rank_certificate(request) == result
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
    "verify_weighted_intersection_rank_certificate",
    "verify_weighted_intersection_result",
    "weighted_intersection_certificate",
    "weighted_intersection_rank_certificate",
]
