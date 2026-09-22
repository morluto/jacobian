from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids._models import (
    LinearMatroid,
    MatroidIntersectionResult,
    MatroidIntersectionWitness,
)
from jacobian.math.combinatorics.matroids.operations import (
    _selected_columns_matrix,
)
from jacobian.math.matrices.finite_fields.linear_algebra import rank as pf_rank

MAX_INTERSECTION_GROUND = 256
# The oracle kernel has O(n^3) exchange probes.  A rank probe is charged for
# the larger representation row count; witness ranks and the result carrier
# are charged separately.  This is deliberately an admission bound, not a
# timeout: every admitted request has a finite exact completion envelope.
MAX_INTERSECTION_WORK = 50_000_000


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


def _admit_work(first: LinearMatroid, second: LinearMatroid) -> None:
    n = first.ground_size
    rows = max(len(first.matrix.entries), len(second.matrix.entries), 1)
    rank_work = (n + 1) ** 3 * rows
    witness_work = (n + 1) ** 2 * rows
    output_work = (n + 1) * 4
    if rank_work + witness_work + output_work > MAX_INTERSECTION_WORK:
        raise OperationResourceAdmissionError(
            location=("first", "second"),
            code="matroid.intersection.work_bound",
            message=(
                "intersection exchange and witness work exceeds the "
                f"{MAX_INTERSECTION_WORK}-unit envelope"
            ),
        )


def _independent(m: LinearMatroid, subset: Sequence[int]) -> bool:
    return len(subset) == pf_rank(_selected_columns_matrix(m, list(subset)))


def matroid_intersection(  # noqa: C901
    first: LinearMatroid, second: LinearMatroid
) -> MatroidIntersectionResult:
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
    n = first.ground_size
    if n > MAX_INTERSECTION_GROUND:
        raise OperationResourceAdmissionError(
            location=("first", "matrix"),
            code="matroid.intersection.work_bound",
            message=f"intersection ground is limited to {MAX_INTERSECTION_GROUND} elements",
        )
    _admit_work(first, second)

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

    def rank(m: LinearMatroid, indices: Sequence[int]) -> int:
        return pf_rank(_selected_columns_matrix(m, list(indices))) if indices else 0

    rank1 = rank(first, subset)
    rank2 = rank(second, tuple(i for i in range(n) if i not in subset))
    if rank1 + rank2 != len(independent_set):
        raise OperationDomainValidationError(
            location=("first", "second"),
            code="matroid.intersection.witness",
            message="exchange kernel did not establish its min-max witness",
        )
    witness = MatroidIntersectionWitness(
        subset=subset,
        rank_first=rank1,
        rank_second_complement=rank2,
        equality=rank1 + rank2,
    )
    return MatroidIntersectionResult(
        first=first,
        second=second,
        common_independent=independent_set,
        cardinality=len(independent_set),
        witness=witness,
    )


__all__ = ["matroid_intersection"]
