"""Exact finite-dimensional algebra operations."""

from jacobian.math.finite_dim_algebras._models import StructureConstants


def _nullspace_mod_prime(
    rows: list[list[int]], dimension: int, prime: int
) -> tuple[tuple[int, ...], ...]:
    matrix = [list(row) for row in rows]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(dimension):
        pivot = next(
            (
                row
                for row in range(pivot_row, len(matrix))
                if matrix[row][column] % prime
            ),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        inverse = pow(matrix[pivot_row][column] % prime, -1, prime)
        matrix[pivot_row] = [(value * inverse) % prime for value in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row == pivot_row or not matrix[row][column] % prime:
                continue
            factor = matrix[row][column] % prime
            matrix[row] = [
                (matrix[row][index] - factor * matrix[pivot_row][index]) % prime
                for index in range(dimension)
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(matrix):
            break

    free_columns = [
        column for column in range(dimension) if column not in pivot_columns
    ]
    basis: list[tuple[int, ...]] = []
    for free_column in free_columns:
        vector = [0] * dimension
        vector[free_column] = 1
        for row, pivot_column in enumerate(pivot_columns):
            vector[pivot_column] = (-matrix[row][free_column]) % prime
        basis.append(tuple(vector))
    return tuple(basis)


def center_basis(algebra: StructureConstants) -> tuple[tuple[int, ...], ...]:
    """Return a canonical basis for the center over the declared prime field."""

    dimension = algebra.dimension
    prime = algebra.field_order
    multiplication = algebra.multiplication
    commutator_rows = [
        [
            (
                multiplication[column][basis][coordinate]
                - multiplication[basis][column][coordinate]
            )
            % prime
            for column in range(dimension)
        ]
        for basis in range(dimension)
        for coordinate in range(dimension)
    ]
    return _nullspace_mod_prime(commutator_rows, dimension, prime)


__all__ = ["center_basis"]
