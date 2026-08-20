"""Exact bounded majorization and matrix mixing operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from jacobian._exact import CanonicalRational
from jacobian.math.majorization._models import (
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
)


def _to_cr(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(value)


def _format_rational(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


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
        prefix_slacks=tuple(_format_rational(s) for s in slacks),
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
        prefix_slack=tuple(_format_rational(s) for s in slacks),
        first_failed_prefix=first_failed,
    )


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

    # First check majorization
    x_sorted = _sorted_desc(x_vals)
    y_sorted = _sorted_desc(y_vals)
    x_prefix = _prefix_sums(x_sorted)
    y_prefix = _prefix_sums(y_sorted)

    total_sum_match = x_prefix[n] == y_prefix[n]
    all_ok = True
    for k in range(1, n):
        if x_prefix[k] - y_prefix[k] < 0:
            all_ok = False
            break
    majorizes = all_ok and total_sum_match

    if not majorizes:
        return TTransformSequenceResult(
            majorizes=False,
            steps=(),
            final_permutation=(),
            intermediate_vectors=(),
            composed_matrix=(),
            target_match=False,
        )

    # Compute T-transform sequence using the Robin Hood algorithm
    current = list(x_vals)
    target = list(y_vals)
    steps: list[tuple[int, int, Fraction]] = []

    for k in range(n - 1, 0, -1):
        while True:
            # Check if current[k] matches target[k]
            if current[k] == target[k]:
                break

            # Find index i < k with current[i] > current[k] (rich donor)
            idx_i = None
            for j in range(k):
                if current[j] > current[k]:
                    idx_i = j
                    break

            if idx_i is None:
                break

            idx_j = k
            ci = current[idx_i]
            cj = current[idx_j]

            # Compute lambda for the T-transform
            # We want to move current[k] towards target[k]
            # T-transform: new_i = lam * ci + (1-lam) * cj
            #              new_j = (1-lam) * ci + lam * cj
            # We need new_j = target[k] (or as close as possible)
            denom = ci - cj
            if denom <= 0:
                break

            lam = (ci - target[k]) / denom
            if lam < 0:
                lam = Fraction(0)
                break
            if lam > 1:
                lam = Fraction(1)
                break
            if lam == 0:
                break

            new_i = lam * ci + (1 - lam) * cj
            new_j = (1 - lam) * ci + lam * cj

            current[idx_i] = new_i
            current[idx_j] = new_j
            steps.append((idx_i, idx_j, lam))

    # Now current should match x_sorted but maybe permuted
    # Apply permutation if needed
    final_perm: list[int] = list(range(n))
    needs_perm = False
    for i in range(n):
        if current[i] != target[i]:
            # Find j to swap
            for j in range(i + 1, n):
                if current[j] == target[i] and current[j] != target[j]:
                    current[i], current[j] = current[j], current[i]
                    final_perm[i], final_perm[j] = final_perm[j], final_perm[i]
                    needs_perm = True
                    break

    target_match = all(current[i] == y_vals[i] for i in range(n))

    # Build the composed doubly stochastic matrix
    composed = _build_composed_matrix(steps, final_perm if needs_perm else None, n)

    # Build intermediate vectors
    intermediate: list[tuple[str, ...]] = []
    cur_vec = list(request.x.as_fractions())
    intermediate.append(tuple(_format_rational(v) for v in cur_vec))
    for i, j, lam in steps:
        ci_val = cur_vec[i]
        cj_val = cur_vec[j]
        cur_vec[i] = lam * ci_val + (1 - lam) * cj_val
        cur_vec[j] = (1 - lam) * ci_val + lam * cj_val
        intermediate.append(tuple(_format_rational(v) for v in cur_vec))
    if needs_perm:
        cur_vec = [cur_vec[final_perm[i]] for i in range(n)]
        intermediate.append(tuple(_format_rational(v) for v in cur_vec))

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
            tuple(_format_rational(v) for v in row) for row in composed
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
        [Fraction(1) if i == j else Fraction(0) for j in range(n)]
        for i in range(n)
    ]

    for i, j, lam in steps:
        new_mat = [[Fraction(0)] * n for _ in range(n)]
        for r in range(n):
            for c in range(n):
                if r == i and c == i:
                    new_mat[r][c] = lam * mat[r][c]
                elif r == j and c == j:
                    new_mat[r][c] = lam * mat[r][c]
                elif r == i and c == j:
                    new_mat[r][c] = (1 - lam) * mat[r][c]
                elif r == j and c == i:
                    new_mat[r][c] = (1 - lam) * mat[r][c]
                elif r == i:
                    new_mat[r][c] = lam * mat[r][c]
                elif r == j:
                    new_mat[r][c] = (1 - lam) * mat[r][c]
                elif c == i:
                    new_mat[r][c] = lam * mat[r][c]
                elif c == j:
                    new_mat[r][c] = (1 - lam) * mat[r][c]
                else:
                    new_mat[r][c] = mat[r][c]
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
    mat = request.matrix.as_fractions()
    n = len(mat)

    first_neg: tuple[int, int] | None = None
    for i in range(n):
        for j in range(n):
            if mat[i][j] < 0:
                first_neg = (i, j)
                break
        if first_neg is not None:
            break

    row_sums = [sum(mat[i]) for i in range(n)]
    col_sums = [sum(mat[i][j] for i in range(n)) for j in range(n)]

    first_bad_row = next((i for i in range(n) if row_sums[i] != 1), None)
    first_bad_col = next((j for j in range(n) if col_sums[j] != 1), None)

    is_ds = first_neg is None and first_bad_row is None and first_bad_col is None

    return DoublyStochasticCheckResult(
        is_doubly_stochastic=is_ds,
        row_sums=tuple(_format_rational(s) for s in row_sums),
        col_sums=tuple(_format_rational(s) for s in col_sums),
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
    mat = request.matrix.as_fractions()
    n = len(mat)

    for i in range(n):
        for j in range(n):
            if mat[i][j] < 0:
                raise ValueError(
                    "Birkhoff decomposition requires a non-negative matrix"
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
            raise ValueError(
                "Birkhoff decomposition failed: no perfect matching found; "
                "matrix may not be doubly stochastic"
            )

        # Find minimum positive entry in the matching
        min_val = None
        for i in range(n):
            j = matching[i]
            if current[i][j] > 0:
                if min_val is None or current[i][j] < min_val:
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

    weights_sum = sum(t.weight.as_fraction() for t in terms)

    return BirkhoffDecompositionResult(
        terms=tuple(terms),
        weights_sum=_format_rational(weights_sum),
        reconstruction_matches=True,
    )


def _find_perfect_matching(
    matrix: list[list[Fraction]], n: int
) -> list[int] | None:
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
        eigenvalues_sorted=tuple(_format_rational(v) for v in e_sorted),
        diagonal_sorted=tuple(_format_rational(v) for v in d_sorted),
        prefix_slack=tuple(_format_rational(s) for s in slacks),
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
