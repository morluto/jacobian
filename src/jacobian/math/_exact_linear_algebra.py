"""Small exact linear-algebra facts shared by mathematical domains."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction


def symmetric_inertia(  # noqa: C901
    matrix: Sequence[Sequence[int | Fraction]],
) -> tuple[int, int, int]:
    """Return positive, negative, and zero inertia of a symmetric matrix.

    Symmetric fraction-free congruence elimination avoids root isolation and
    computes inertia directly by Sylvester's law. A zero diagonal with a
    nonzero off-diagonal entry contributes one positive and one negative
    direction; otherwise the remaining zero block is the radical.
    """

    values = [[Fraction(value) for value in row] for row in matrix]
    dimension = len(values)
    if any(len(row) != dimension for row in values):
        raise ValueError("symmetric inertia requires a square matrix")
    if any(
        values[row][column] != values[column][row]
        for row in range(dimension)
        for column in range(row)
    ):
        raise ValueError("symmetric inertia requires a symmetric matrix")

    positive = negative = zero = 0
    index = 0
    while index < dimension:
        pivot = next((row for row in range(index, dimension) if values[row][row]), None)
        if pivot is not None:
            if pivot != index:
                values[index], values[pivot] = values[pivot], values[index]
                for row in range(dimension):
                    values[row][index], values[row][pivot] = (
                        values[row][pivot],
                        values[row][index],
                    )
            diagonal = values[index][index]
            positive += diagonal > 0
            negative += diagonal < 0
            for row in range(index + 1, dimension):
                for column in range(row, dimension):
                    values[row][column] -= (
                        values[row][index] * values[column][index] / diagonal
                    )
                    values[column][row] = values[row][column]
            index += 1
            continue

        pair = next(
            (
                (row, column)
                for row in range(index, dimension)
                for column in range(row + 1, dimension)
                if values[row][column]
            ),
            None,
        )
        if pair is None:
            zero += dimension - index
            break
        first, second = pair
        for target, source in ((index, first), (index + 1, second)):
            if source != target:
                values[target], values[source] = values[source], values[target]
                for row in range(dimension):
                    values[row][target], values[row][source] = (
                        values[row][source],
                        values[row][target],
                    )
                if first == target:
                    first = source
                if second == target:
                    second = source
        # The 2x2 pivot has zero diagonal and nonzero off-diagonal, hence
        # determinant < 0 and inertia (1, 1, 0). Its Schur complement is exact.
        off_diagonal = values[index][index + 1]
        positive += 1
        negative += 1
        for row in range(index + 2, dimension):
            for column in range(row, dimension):
                values[row][column] -= (
                    values[row][index] * values[column][index + 1]
                    + values[row][index + 1] * values[column][index]
                ) / off_diagonal
                values[column][row] = values[row][column]
        index += 2
    return positive, negative, zero


__all__ = ["symmetric_inertia"]
