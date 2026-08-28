"""Exact native kernels over combinatorial sign and Hadamard matrices."""

from __future__ import annotations

from ._models import (
    DeterminantProfileResult,
    GramProfileResult,
    KroneckerProductResult,
    NormalizeResult,
    SignProfileResult,
    SylvesterResult,
)
from .values import MAX_MATRIX_ORDER, HadamardMatrix, SignMatrix

__all__ = [
    "determinant_profile",
    "gram_profile",
    "kronecker",
    "normalize",
    "sign_profile",
    "sylvester",
]


def sign_profile(matrix: SignMatrix) -> SignProfileResult:
    """Return dimensions, entry counts, row/column sums, and first/all
    non-sign entries for a general integer matrix."""
    row_count = len(matrix.rows)
    col_count = len(matrix.rows[0]) if row_count else 0
    row_sums = [sum(row) for row in matrix.rows]
    col_sums = [
        sum(matrix.rows[i][j] for i in range(row_count)) for j in range(col_count)
    ]
    plus_one = sum(1 for row in matrix.rows for entry in row if entry == 1)
    minus_one = sum(1 for row in matrix.rows for entry in row if entry == -1)
    return SignProfileResult(
        row_count=row_count,
        column_count=col_count,
        plus_one_count=plus_one,
        minus_one_count=minus_one,
        row_sums=tuple(row_sums),
        column_sums=tuple(col_sums),
        is_square=row_count == col_count,
    )


def gram_profile(matrix: SignMatrix) -> GramProfileResult:
    """Return order, exact ``H H^T``, diagonal residuals from n, all nonzero
    off-diagonal inner products, and ``is_hadamard``."""
    rows = matrix.rows
    n = len(rows)
    m = len(rows[0]) if n else 0
    gram: list[list[int]] = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            inner = sum(rows[i][k] * rows[j][k] for k in range(m))
            gram[i][j] = inner
            gram[j][i] = inner
    is_hadamard = n == m and all(
        gram[i][j] == (n if i == j else 0) for i in range(n) for j in range(n)
    )
    residuals = tuple(gram[i][i] - m for i in range(n))
    nonzero_off = tuple(
        (i, j, gram[i][j]) for i in range(n) for j in range(i + 1, n) if gram[i][j] != 0
    )
    return GramProfileResult(
        order=n,
        gram=tuple(tuple(row) for row in gram),
        diagonal_residuals=residuals,
        nonzero_off_diagonal=nonzero_off,
        is_hadamard=is_hadamard,
    )


def normalize(matrix: HadamardMatrix | SignMatrix) -> NormalizeResult:
    """Return a deterministically normalized sign matrix whose first row and
    first column are all ``+1``, plus the exact row/column sign switches
    used. Normalization must preserve the full matrix and be idempotent."""
    rows = [list(row) for row in matrix.rows]
    row_switches: list[int] = [0] * len(rows)
    col_switches: list[int] = [0] * (len(rows[0]) if rows else 0)
    for j in range(len(rows[0])):
        if rows[0][j] == -1:
            col_switches[j] = 1
            for i in range(len(rows)):
                rows[i][j] = -rows[i][j]
    for i in range(len(rows)):
        if rows[i][0] == -1:
            row_switches[i] = 1
            for j in range(len(rows[0])):
                rows[i][j] = -rows[i][j]
    value_type = HadamardMatrix if isinstance(matrix, HadamardMatrix) else SignMatrix
    return NormalizeResult(
        normalized=value_type(rows=tuple(tuple(row) for row in rows)),
        row_switches=tuple(row_switches),
        column_switches=tuple(col_switches),
    )


def determinant_profile(hadamard: HadamardMatrix) -> DeterminantProfileResult:
    """For a constructed Hadamard matrix of order n, return |det H| = n^(n/2)
    and the Gram determinant = n^n."""
    n = len(hadamard.rows)
    if n % 2 != 0 and n != 1:
        raise ValueError("Hadamard matrices have even order (except order 1)")
    magnitude = n ** (n // 2)
    gram_determinant = n**n
    return DeterminantProfileResult(
        order=n,
        determinant_magnitude=magnitude,
        gram_determinant=gram_determinant,
        identity="det(H)^2 = det(H H^T)",
    )


def kronecker(left: HadamardMatrix, right: HadamardMatrix) -> KroneckerProductResult:
    """Return the Kronecker product of two Hadamard matrices as a Hadamard
    matrix, factor-to-product row/column maps, and the exact Gram
    factorization."""
    a = [list(row) for row in left.rows]
    b = [list(row) for row in right.rows]
    n, m = len(a), len(b)
    if n * m > MAX_MATRIX_ORDER:
        raise ValueError(
            f"Kronecker product order {n * m} exceeds maximum {MAX_MATRIX_ORDER}"
        )
    result: list[list[int]] = []
    row_map: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(m):
            new_row: list[int] = []
            for ai in range(n):
                for bj in range(m):
                    new_row.append(a[i][ai] * b[j][bj])
            result.append(new_row)
            row_map.append((i, j))
    col_map: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(m):
            col_map.append((i, j))
    return KroneckerProductResult(
        product=HadamardMatrix(rows=tuple(tuple(row) for row in result)),
        row_map=tuple(row_map),
        column_map=tuple(col_map),
    )


def sylvester(k: int) -> SylvesterResult:
    """For bounded ``k``, return the recursively defined order ``2^k``
    Hadamard matrix with construction ledger."""
    if k < 0 or k > 7:
        raise ValueError("k must be in [0, 7]")
    if k == 0:
        return SylvesterResult(
            matrix=HadamardMatrix(rows=((1,),)),
            construction="base_case",
            order=1,
        )
    prev_result = sylvester(k - 1)
    prev = [list(row) for row in prev_result.matrix.rows]
    n = len(prev)
    top = [prev[i] + prev[i] for i in range(n)]
    bottom = [prev[i] + [-prev[i][j] for j in range(n)] for i in range(n)]
    result = top + bottom
    return SylvesterResult(
        matrix=HadamardMatrix(rows=tuple(tuple(row) for row in result)),
        construction="sylvester_recursion",
        order=2**k,
    )
