"""Domain functions for linear code structural operations."""

from __future__ import annotations

from jacobian.math.code_linear._models import (
    DualCodeResult,
    GeneratorMatrixRequest,
    MacWilliamsRequest,
    MacWilliamsResult,
    PunctureRequest,
    PunctureResult,
)


def _rref(matrix: list[list[int]], field_order: int) -> tuple[list[list[int]], int]:
    rows = [list(row) for row in matrix]
    row_count = len(rows)
    col_count = len(rows[0])
    pivot_row = 0
    for col in range(col_count):
        pivot = None
        for i in range(pivot_row, row_count):
            if rows[i][col] % field_order != 0:
                pivot = i
                break
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        inv = pow(rows[pivot_row][col] % field_order, -1, field_order)
        rows[pivot_row] = [v * inv % field_order for v in rows[pivot_row]]
        for i, row in enumerate(rows):
            if i == pivot_row:
                continue
            factor = row[col] % field_order
            if factor == 0:
                continue
            rows[i] = [
                (a - factor * b) % field_order
                for a, b in zip(row, rows[pivot_row], strict=True)
            ]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return rows, pivot_row


def _nullspace(matrix: list[list[int]], field_order: int) -> list[list[int]]:
    rows, rank = _rref(matrix, field_order)
    n = len(matrix[0])
    piv_cols: list[int] = []
    for i in range(rank):
        piv = n
        for j in range(n):
            if rows[i][j] % field_order != 0:
                piv = j
                break
        piv_cols.append(piv)
    free_cols = [j for j in range(n) if j not in piv_cols]
    basis: list[list[int]] = []
    for fc in free_cols:
        vec = [0] * n
        vec[fc] = 1
        for i in range(rank):
            piv = piv_cols[i]
            vec[piv] = (-rows[i][fc]) % field_order
        basis.append(vec)
    return basis


def compute_dual_code(request: GeneratorMatrixRequest) -> DualCodeResult:
    matrix = [list(row) for row in request.generator_matrix]
    _, rank = _rref(matrix, request.field_order)
    null = _nullspace(matrix, request.field_order)
    length = len(request.generator_matrix[0])
    return DualCodeResult(
        dual_generator=tuple(tuple(row) for row in null),
        dimension=rank,
        dual_dimension=length - rank,
        length=length,
    )


def compute_macwilliams_transform(request: MacWilliamsRequest) -> MacWilliamsResult:
    from math import comb

    q = request.field_order
    primal = list(request.weights)
    n = len(primal) - 1
    dual = []
    for k in range(n + 1):
        s = 0
        for i in range(n + 1):
            for j in range(i + 1):
                if j > k or i - j > n - j:
                    continue
                term = comb(i, j) * comb(n - j, k - j) * (-1) ** j * (q - 1) ** (i - j)
                s += primal[i] * term
        dual.append(s // request.code_cardinality)
    return MacWilliamsResult(dual_weights=tuple(dual))


def compute_puncture(request: PunctureRequest) -> PunctureResult:
    matrix = [list(row) for row in request.generator_matrix]
    punctured = [
        row[: request.coordinate] + row[request.coordinate + 1 :] for row in matrix
    ]
    rows, rank = _rref(punctured, request.field_order)
    new_len = len(matrix[0]) - 1
    gen = tuple(tuple(row) for row in rows[:rank]) if rank > 0 else ((0,) * new_len,)
    return PunctureResult(
        generator=gen,
        dimension=rank,
        length=new_len,
    )
