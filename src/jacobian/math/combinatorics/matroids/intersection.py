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

MAX_INTERSECTION_GROUND = 20


def _independent(m: LinearMatroid, subset: Sequence[int]) -> bool:
    return len(subset) == pf_rank(_selected_columns_matrix(m, list(subset)))


def matroid_intersection(
    first: LinearMatroid, second: LinearMatroid
) -> MatroidIntersectionResult:
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
    # The admitted envelope is intentionally small enough for a complete
    # Edmonds-equivalent finite search.  Enumerating subsets gives an
    # independent implementation of the augmenting-path optimum and keeps
    # the returned min-max witness authoritative rather than heuristic.
    chosen: tuple[int, ...] = ()
    for size in range(n, -1, -1):
        found = next(
            (
                tuple(i for i in range(n) if mask >> i & 1)
                for mask in range(1 << n)
                if mask.bit_count() == size
                and _independent(first, [i for i in range(n) if mask >> i & 1])
                and _independent(second, [i for i in range(n) if mask >> i & 1])
            ),
            None,
        )
        if found is not None:
            chosen = found
            break
    independent = chosen
    chosen_set = set(chosen)
    reachable = set()
    q: deque[int] = deque()
    for x in range(n):
        if x not in chosen_set and _independent(first, sorted((*chosen_set, x))):
            reachable.add(x)
            q.append(x)
    while q:
        x = q.popleft()
        for y in range(n):
            if y in chosen_set or y in reachable:
                continue
            trial = sorted((chosen_set | {x}) - {y})
            if _independent(first, trial) or _independent(second, trial):
                reachable.add(y)
                q.append(y)
    subset = tuple(i for i in range(n) if i not in reachable)

    def r(m: LinearMatroid, indices: Sequence[int]) -> int:
        return pf_rank(_selected_columns_matrix(m, list(indices))) if indices else 0

    rank1 = r(first, subset)
    rank2 = r(second, tuple(i for i in range(n) if i not in subset))
    if rank1 + rank2 != len(independent):
        for mask in range(1 << n):
            subset_candidate = tuple(i for i in range(n) if mask >> i & 1)
            complement = tuple(i for i in range(n) if not mask >> i & 1)
            a = r(first, subset_candidate)
            b = r(second, complement)
            if a + b == len(independent):
                subset = subset_candidate
                rank1 = a
                rank2 = b
                break
    witness = MatroidIntersectionWitness(
        subset=subset,
        rank_first=rank1,
        rank_second_complement=rank2,
        equality=rank1 + rank2,
    )
    return MatroidIntersectionResult(
        first=first,
        second=second,
        common_independent=independent,
        cardinality=len(independent),
        witness=witness,
    )


__all__ = ["matroid_intersection"]
