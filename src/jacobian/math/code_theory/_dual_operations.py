"""Dual code and syndrome operations for coding theory."""

from jacobian.math.code_theory._models import (
    DualCodeRequest,
    DualCodeResult,
    SyndromeRequest,
    SyndromeResult,
)


def _nullspace_mod_prime(rows: tuple[tuple[int, ...], ...], p: int) -> list[list[int]]:
    """Compute the nullspace of a matrix over GF(p) via Gaussian elimination."""
    row_count = len(rows)
    col_count = len(rows[0]) if row_count > 0 else 0
    mat = [list(row) for row in rows]

    pivot_row = 0
    pivot_cols: list[int] = []
    for col in range(col_count):
        pivot = next(
            (i for i in range(pivot_row, row_count) if mat[i][col] % p != 0),
            None,
        )
        if pivot is None:
            continue
        mat[pivot_row], mat[pivot] = mat[pivot], mat[pivot_row]
        inv = pow(mat[pivot_row][col] % p, -1, p)
        mat[pivot_row] = [v * inv % p for v in mat[pivot_row]]
        for i in range(row_count):
            if i == pivot_row:
                continue
            factor = mat[i][col] % p
            if factor == 0:
                continue
            mat[i] = [
                (a - factor * b) % p
                for a, b in zip(mat[i], mat[pivot_row], strict=True)
            ]
        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == row_count:
            break

    # Free columns correspond to nullspace basis vectors
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(col_count) if c not in pivot_set]

    null_basis: list[list[int]] = []
    for fc in free_cols:
        vec = [0] * col_count
        vec[fc] = 1
        for ri, pc in enumerate(pivot_cols):
            vec[pc] = (-mat[ri][fc]) % p
        null_basis.append(vec)

    return null_basis


def compute_dual_code(request: DualCodeRequest) -> DualCodeResult:
    """Compute the dual code (parity check matrix) from a generator matrix.

    Uses Gaussian elimination over GF(p) to find the parity
    check matrix H such that G * H^T = 0.
    """
    from jacobian.math.code_theory._models import _matrix_rank_mod_prime

    p = request.field_order
    rows = request.generator_matrix
    n = len(rows[0])

    rank = _matrix_rank_mod_prime(request.generator_matrix, p)
    null_basis = _nullspace_mod_prime(request.generator_matrix, p)

    parity_check: tuple[tuple[int, ...], ...] = tuple(tuple(vec) for vec in null_basis)

    return DualCodeResult(
        field_order=p,
        parity_check_matrix=parity_check,
        code_dimension=rank,
        code_length=n,
        dual_dimension=len(parity_check),
    )


def compute_syndrome(request: SyndromeRequest) -> SyndromeResult:
    """Compute the syndrome H * r^T mod p for a received word."""
    p = request.field_order
    h = request.parity_check_matrix
    r = request.received_word
    num_rows = len(h)
    num_cols = len(r)

    syndrome = []
    for i in range(num_rows):
        s = sum(h[i][j] * r[j] for j in range(num_cols)) % p
        syndrome.append(s)

    return SyndromeResult(
        field_order=p,
        syndrome=tuple(syndrome),
    )
