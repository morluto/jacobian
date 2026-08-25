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
    col_count = len(rows[0]) if rows else 0
    matrix = [list(row) for row in rows]

    pivot_row = 0
    pivot_columns: list[int] = []
    for column in range(col_count):
        pivot = next(
            (row for row in range(pivot_row, row_count) if matrix[row][column] % p),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        inverse = pow(matrix[pivot_row][column] % p, -1, p)
        matrix[pivot_row] = [value * inverse % p for value in matrix[pivot_row]]
        for row in range(row_count):
            if row == pivot_row:
                continue
            factor = matrix[row][column] % p
            if factor:
                matrix[row] = [
                    (left - factor * right) % p
                    for left, right in zip(matrix[row], matrix[pivot_row], strict=True)
                ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == row_count:
            break

    pivot_set = set(pivot_columns)
    null_basis: list[list[int]] = []
    for free_column in (
        column for column in range(col_count) if column not in pivot_set
    ):
        vector = [0] * col_count
        vector[free_column] = 1
        for row, pivot_column in enumerate(pivot_columns):
            vector[pivot_column] = (-matrix[row][free_column]) % p
        null_basis.append(vector)
    return null_basis


def compute_dual_code(request: DualCodeRequest) -> DualCodeResult:
    """Compute the parity-check matrix spanning a code's dual over GF(p)."""
    from jacobian.math.code_theory._models import _matrix_rank_mod_prime

    field_order = request.field_order
    code_dimension = _matrix_rank_mod_prime(request.generator_matrix, field_order)
    parity_check_matrix = tuple(
        tuple(vector)
        for vector in _nullspace_mod_prime(request.generator_matrix, field_order)
    )
    return DualCodeResult(
        field_order=field_order,
        parity_check_matrix=parity_check_matrix,
        code_dimension=code_dimension,
        code_length=len(request.generator_matrix[0]),
        dual_dimension=len(parity_check_matrix),
    )


def compute_syndrome(request: SyndromeRequest) -> SyndromeResult:
    """Compute the syndrome H * r^T mod p for a received word."""
    return SyndromeResult(
        field_order=request.field_order,
        syndrome=tuple(
            sum(
                entry * coordinate
                for entry, coordinate in zip(row, request.received_word, strict=True)
            )
            % request.field_order
            for row in request.parity_check_matrix
        ),
    )
