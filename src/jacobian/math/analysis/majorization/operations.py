"""Exact bounded majorization and matrix mixing operations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, format_canonical_rational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis.majorization._models import (
    BirkhoffDecompositionRequest,
    BirkhoffDecompositionResult,
    BirkhoffTerm,
    DoublyStochasticCheckRequest,
    DoublyStochasticCheckResult,
    MajorizationCheckRequest,
    MajorizationCheckResult,
    SchurHornCheckRequest,
    SchurHornCheckResult,
    TTransformSequenceRequest,
    TTransformSequenceResult,
    TTransformStep,
    WeakMajorizationCheckRequest,
    WeakMajorizationCheckResult,
    _bound_rational,
    _require_majorization_matrix,
)
from jacobian.math.matrices.values import RationalMatrix


def _run_admission(
    admission: Callable[[], None], *, location: tuple[str | int, ...]
) -> None:
    """Expose owner admission failures through the native operation contract."""

    try:
        admission()
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc


def _admit_bounded_rational(
    value: CanonicalRational,
    *,
    label: str,
    location: tuple[str | int, ...],
) -> None:
    """Apply the majorization rational envelope at a field-specific path."""

    try:
        _bound_rational(value, label)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=location, code="majorization.admission", message=str(exc)
        ) from exc


def _to_cr(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(value)


def _matrix_fractions(matrix: RationalMatrix) -> list[list[Fraction]]:
    return [[value.as_fraction() for value in row] for row in matrix.entries]


def _sorted_desc(values: Sequence[Fraction]) -> list[Fraction]:
    return sorted(values, reverse=True)


def _prefix_sums(values: list[Fraction]) -> list[Fraction]:
    """Compute cumulative prefix sums: prefix[k] = sum(values[:k])."""
    sums = [Fraction(0)] * (len(values) + 1)
    for i, v in enumerate(values):
        sums[i + 1] = sums[i] + v
    return sums


def compute_majorization_check(
    request: MajorizationCheckRequest,
) -> MajorizationCheckResult:
    """Check if x majorizes y (ordinary majorization).

    x majorizes y when, after sorting both in nonincreasing order:
    - sum_{i=1}^k x_i >= sum_{i=1}^k y_i for all 1 <= k < n
    - sum_{i=1}^n x_i = sum_{i=1}^n y_i
    """
    x_vals = request.x.as_fractions()
    y_vals = request.y.as_fractions()
    n = len(x_vals)

    x_sorted = _sorted_desc(x_vals)
    y_sorted = _sorted_desc(y_vals)

    x_prefix = _prefix_sums(x_sorted)
    y_prefix = _prefix_sums(y_sorted)

    total_sum_match = x_prefix[n] == y_prefix[n]

    slacks: list[Fraction] = []
    first_failed: int | None = None
    all_ok = True

    for k in range(1, n):
        slack = x_prefix[k] - y_prefix[k]
        slacks.append(slack)
        if slack < 0:
            all_ok = False
            if first_failed is None:
                first_failed = k

    majorizes = all_ok and total_sum_match

    return MajorizationCheckResult(
        majorizes=majorizes,
        total_sum_match=total_sum_match,
        prefix_slacks=tuple(format_canonical_rational(s) for s in slacks),
        first_failed_prefix=first_failed,
    )


def compute_weak_majorization_check(
    request: WeakMajorizationCheckRequest,
) -> WeakMajorizationCheckResult:
    """Check weak majorization.

    For 'sub' (weak submajorization):
    sum_{i=1}^k x_i^down >= sum_{i=1}^k y_i^down for all 1 <= k <= n

    For 'super' (weak supermajorization):
    sum_{i=1}^k x_i^up <= sum_{i=1}^k y_i^up for all 1 <= k <= n
    (using ascending sort)
    """
    x_vals = request.x.as_fractions()
    y_vals = request.y.as_fractions()
    n = len(x_vals)
    direction = request.direction

    if direction == "sub":
        x_sorted = _sorted_desc(x_vals)
        y_sorted = _sorted_desc(y_vals)
    else:
        x_sorted = sorted(x_vals)
        y_sorted = sorted(y_vals)

    x_prefix = _prefix_sums(x_sorted)
    y_prefix = _prefix_sums(y_sorted)

    slacks: list[Fraction] = []
    first_failed: int | None = None
    all_ok = True

    for k in range(1, n + 1):
        if direction == "sub":
            slack = x_prefix[k] - y_prefix[k]
        else:
            slack = y_prefix[k] - x_prefix[k]
        slacks.append(slack)
        if slack < 0:
            all_ok = False
            if first_failed is None:
                first_failed = k

    return WeakMajorizationCheckResult(
        holds=all_ok,
        direction=direction,
        prefix_slack=tuple(format_canonical_rational(s) for s in slacks),
        first_failed_prefix=first_failed,
    )


def _majorizes_values(x_vals: Sequence[Fraction], y_vals: Sequence[Fraction]) -> bool:
    n = len(x_vals)
    x_prefix = _prefix_sums(_sorted_desc(x_vals))
    y_prefix = _prefix_sums(_sorted_desc(y_vals))
    if x_prefix[n] != y_prefix[n]:
        return False
    return all(x_prefix[k] >= y_prefix[k] for k in range(1, n))


def _compute_t_transform_steps(
    x_vals: list[Fraction], target: list[Fraction], order: list[int]
) -> tuple[list[Fraction], list[tuple[int, int, Fraction]]]:
    """Use the rank-aligned Hardy--Littlewood--Pólya construction.

    ``order`` places the original coordinates in nonincreasing source order.
    The target is assigned in that order, so every mixing step acts on the
    caller's original coordinates while preserving the theorem's sorted-vector
    invariant.  A final permutation then restores the requested target labels.
    """

    n = len(x_vals)
    current = list(x_vals)
    steps: list[tuple[int, int, Fraction]] = []
    for _ in range(n - 1):
        if all(current[index] == target[index] for index in order):
            break
        i_rank = next(
            (
                rank
                for rank, index in enumerate(order)
                if current[index] > target[index]
            ),
            None,
        )
        if i_rank is None:
            break
        i_idx = order[i_rank]
        ci = current[i_idx]
        target_i = target[i_idx]
        j_rank = next(
            (rank for rank in range(i_rank + 1, n) if current[order[rank]] <= target_i),
            None,
        )
        if j_rank is None:
            break
        j_idx = order[j_rank]
        cj = current[j_idx]
        denom = ci - cj
        if denom == 0:
            break
        lam = (target_i - cj) / denom
        if lam < 0 or lam >= 1:
            break
        current[i_idx] = lam * ci + (Fraction(1) - lam) * cj
        current[j_idx] = (Fraction(1) - lam) * ci + lam * cj
        steps.append((i_idx, j_idx, lam))
    return current, steps


def _target_permutation(
    current: list[Fraction], target: list[Fraction]
) -> tuple[list[int], bool, list[Fraction]]:
    n = len(current)
    final_perm = list(range(n))
    if current == target or sorted(current) != sorted(target):
        return final_perm, False, current
    used = [False] * n
    perm = [0] * n
    for idx in range(n):
        found = next(
            (j for j in range(n) if not used[j] and current[j] == target[idx]), None
        )
        if found is None:
            return final_perm, False, current
        perm[idx] = found
        used[found] = True
    if not all(used):
        return final_perm, False, current
    reordered = [current[perm[i]] for i in range(n)]
    if reordered != target:
        return final_perm, False, current
    return perm, True, reordered


def _intermediate_vectors(
    x_vals: list[Fraction],
    steps: list[tuple[int, int, Fraction]],
    final_perm: list[int],
    needs_perm: bool,
) -> list[tuple[str, ...]]:
    intermediate: list[tuple[str, ...]] = []
    current = list(x_vals)
    intermediate.append(tuple(format_canonical_rational(v) for v in current))
    for i, j, lam in steps:
        ci_val = current[i]
        cj_val = current[j]
        current[i] = lam * ci_val + (Fraction(1) - lam) * cj_val
        current[j] = (Fraction(1) - lam) * ci_val + lam * cj_val
        intermediate.append(tuple(format_canonical_rational(v) for v in current))
    if needs_perm:
        current = [current[final_perm[i]] for i in range(len(current))]
        intermediate.append(tuple(format_canonical_rational(v) for v in current))
    return intermediate


def compute_t_transform_sequence(
    request: TTransformSequenceRequest,
) -> TTransformSequenceResult:
    """Compute an exact T-transform sequence from x to y.

    If x majorizes y, returns a sequence of T-transforms (and optionally a
    permutation) such that y = D * x where D is the composed doubly stochastic matrix.
    If x does not majorize y, returns a negative result.
    """
    x_vals = list(request.x.as_fractions())
    y_vals = list(request.y.as_fractions())
    labels = list(request.x.labels)
    n = len(x_vals)

    if not _majorizes_values(x_vals, y_vals):
        return TTransformSequenceResult(
            majorizes=False,
            steps=(),
            final_permutation=(),
            intermediate_vectors=(),
            composed_matrix=(),
            target_match=False,
        )

    order = sorted(range(n), key=lambda index: (-x_vals[index], index))
    sorted_target = _sorted_desc(y_vals)
    rank_aligned_target = [Fraction(0)] * n
    for rank, index in enumerate(order):
        rank_aligned_target[index] = sorted_target[rank]

    current, steps = _compute_t_transform_steps(x_vals, rank_aligned_target, order)
    final_perm, needs_perm, current = _target_permutation(current, list(y_vals))

    target_match = current == y_vals

    # Build the composed doubly stochastic matrix
    composed = _build_composed_matrix(steps, final_perm if needs_perm else None, n)

    intermediate = _intermediate_vectors(x_vals, steps, final_perm, needs_perm)

    step_objs = tuple(
        TTransformStep(
            i_label=labels[i],
            j_label=labels[j],
            lam=_to_cr(lam),
        )
        for i, j, lam in steps
    )

    return TTransformSequenceResult(
        majorizes=True,
        steps=step_objs,
        final_permutation=tuple(final_perm) if needs_perm else (),
        intermediate_vectors=tuple(intermediate),
        composed_matrix=tuple(
            tuple(format_canonical_rational(v) for v in row) for row in composed
        ),
        target_match=target_match,
    )


def _build_composed_matrix(
    steps: list[tuple[int, int, Fraction]],
    final_perm: list[int] | None,
    n: int,
) -> list[list[Fraction]]:
    """Build the composed doubly stochastic matrix from T-transform steps."""
    mat: list[list[Fraction]] = [
        [Fraction(1) if i == j else Fraction(0) for j in range(n)] for i in range(n)
    ]

    for i, j, lam in steps:
        # Left-multiply by T_{i,j}(lam): mixes rows i and j
        new_mat = [row[:] for row in mat]
        for c in range(n):
            orig_i = mat[i][c]
            orig_j = mat[j][c]
            new_mat[i][c] = lam * orig_i + (Fraction(1) - lam) * orig_j
            new_mat[j][c] = (Fraction(1) - lam) * orig_i + lam * orig_j
        mat = new_mat

    if final_perm is not None:
        perm_mat = [[Fraction(0)] * n for _ in range(n)]
        for r in range(n):
            perm_mat[r][final_perm[r]] = Fraction(1)
        result = [[Fraction(0)] * n for _ in range(n)]
        for r in range(n):
            for c in range(n):
                for k in range(n):
                    result[r][c] += perm_mat[r][k] * mat[k][c]
        mat = result

    return mat


def compute_doubly_stochastic_check(
    request: DoublyStochasticCheckRequest,
) -> DoublyStochasticCheckResult:
    """Check if a rational matrix is doubly stochastic."""
    _run_admission(
        lambda: _require_majorization_matrix(request.matrix),
        location=("matrix",),
    )
    mat = _matrix_fractions(request.matrix)
    n = len(mat)

    first_neg: tuple[int, int] | None = None
    for i in range(n):
        for j in range(n):
            if mat[i][j] < 0:
                first_neg = (i, j)
                break
        if first_neg is not None:
            break

    row_sums = [sum(mat[i], Fraction(0)) for i in range(n)]
    col_sums = [sum((mat[i][j] for i in range(n)), Fraction(0)) for j in range(n)]

    first_bad_row = next((i for i in range(n) if row_sums[i] != 1), None)
    first_bad_col = next((j for j in range(n) if col_sums[j] != 1), None)

    is_ds = first_neg is None and first_bad_row is None and first_bad_col is None

    return DoublyStochasticCheckResult(
        is_doubly_stochastic=is_ds,
        row_sums=tuple(format_canonical_rational(s) for s in row_sums),
        col_sums=tuple(format_canonical_rational(s) for s in col_sums),
        first_negative_entry=first_neg,
        first_bad_row=first_bad_row,
        first_bad_col=first_bad_col,
    )


def compute_birkhoff_decomposition(
    request: BirkhoffDecompositionRequest,
) -> BirkhoffDecompositionResult:
    """Compute a Birkhoff-von Neumann decomposition of a doubly stochastic matrix.

    Decomposes a doubly stochastic matrix into a convex combination of
    permutation matrices using the greedy matching + peel algorithm.
    """
    _run_admission(
        lambda: _require_majorization_matrix(request.matrix),
        location=("matrix",),
    )
    mat = _matrix_fractions(request.matrix)
    n = len(mat)

    for i in range(n):
        for j in range(n):
            if mat[i][j] < 0:
                raise OperationDomainValidationError(
                    location=("matrix", "entries", i, j),
                    code="majorization.birkhoff_negative_entry",
                    message="Birkhoff decomposition requires a non-negative matrix",
                )
    for i, row in enumerate(mat):
        if sum(row, Fraction()) != 1:
            raise OperationDomainValidationError(
                location=("matrix", "entries", i),
                code="majorization.birkhoff_row_sum",
                message="Birkhoff decomposition requires row sums equal to 1",
            )
    for j in range(n):
        if sum((mat[i][j] for i in range(n)), Fraction()) != 1:
            raise OperationDomainValidationError(
                location=("matrix", "entries", j),
                code="majorization.birkhoff_column_sum",
                message="Birkhoff decomposition requires column sums equal to 1",
            )

    current = [list(row) for row in mat]
    terms: list[BirkhoffTerm] = []

    while True:
        # Check if all zeros
        if all(all(current[i][j] == 0 for j in range(n)) for i in range(n)):
            break

        # Find perfect matching in the bipartite graph of positive entries
        matching = _find_perfect_matching(current, n)
        if matching is None:
            raise RuntimeError(
                "Birkhoff decomposition failed: no perfect matching found; "
                "matrix may not be doubly stochastic"
            )

        # Find minimum positive entry in the matching
        min_val = None
        for i in range(n):
            j = matching[i]
            if current[i][j] > 0 and (min_val is None or current[i][j] < min_val):
                min_val = current[i][j]

        if min_val is None or min_val <= 0:
            break

        # Subtract the matching
        for i in range(n):
            j = matching[i]
            current[i][j] -= min_val

        terms.append(
            BirkhoffTerm(
                weight=_to_cr(min_val),
                permutation=tuple(matching),
            )
        )

    weights_sum = sum((t.weight.as_fraction() for t in terms), Fraction(0))

    return BirkhoffDecompositionResult(
        terms=tuple(terms),
        weights_sum=format_canonical_rational(weights_sum),
        reconstruction_matches=True,
    )


def _find_perfect_matching(matrix: list[list[Fraction]], n: int) -> list[int] | None:
    """Find a perfect matching in the bipartite graph of positive entries."""
    match_col = [-1] * n
    match_row = [-1] * n

    def augment(u: int, visited: list[bool]) -> bool:
        for v in range(n):
            if matrix[u][v] > 0 and not visited[v]:
                visited[v] = True
                if match_row[v] == -1 or augment(match_row[v], visited):
                    match_col[u] = v
                    match_row[v] = u
                    return True
        return False

    for u in range(n):
        visited = [False] * n
        if not augment(u, visited):
            return None

    return match_col


def compute_schur_horn_check(
    request: SchurHornCheckRequest,
) -> SchurHornCheckResult:
    """Check Schur-Horn feasibility.

    A diagonal vector d is realizable as the diagonal of a Hermitian matrix
    with eigenvalues lambda iff lambda majorizes d.
    """
    for index, value in enumerate(request.eigenvalues):
        _admit_bounded_rational(
            value,
            label=f"eigenvalues[{index}]",
            location=("eigenvalues", index),
        )
    for index, value in enumerate(request.diagonal):
        _admit_bounded_rational(
            value,
            label=f"diagonal[{index}]",
            location=("diagonal", index),
        )
    eigenvalues = [v.as_fraction() for v in request.eigenvalues]
    diagonal = [v.as_fraction() for v in request.diagonal]

    e_sorted = _sorted_desc(eigenvalues)
    d_sorted = _sorted_desc(diagonal)
    n = len(eigenvalues)

    e_prefix = _prefix_sums(e_sorted)
    d_prefix = _prefix_sums(d_sorted)

    total_sum_match = e_prefix[n] == d_prefix[n]

    slacks: list[Fraction] = []
    first_failed: int | None = None
    all_ok = True

    for k in range(1, n):
        slack = e_prefix[k] - d_prefix[k]
        slacks.append(slack)
        if slack < 0:
            all_ok = False
            if first_failed is None:
                first_failed = k

    feasible = all_ok and total_sum_match

    return SchurHornCheckResult(
        feasible=feasible,
        eigenvalues_sorted=tuple(format_canonical_rational(v) for v in e_sorted),
        diagonal_sorted=tuple(format_canonical_rational(v) for v in d_sorted),
        prefix_slack=tuple(format_canonical_rational(s) for s in slacks),
        first_failed_prefix=first_failed,
        total_sum_match=total_sum_match,
    )


__all__ = [
    "compute_birkhoff_decomposition",
    "compute_doubly_stochastic_check",
    "compute_majorization_check",
    "compute_schur_horn_check",
    "compute_t_transform_sequence",
    "compute_weak_majorization_check",
]
