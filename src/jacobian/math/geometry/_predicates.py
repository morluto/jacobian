"""Exact predicates shared by planar-geometry contracts and operations."""

from fractions import Fraction


def are_collinear(
    a: tuple[Fraction, Fraction],
    b: tuple[Fraction, Fraction],
    c: tuple[Fraction, Fraction],
) -> bool:
    """Return whether three exact planar points lie on one line."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0


def determinant3(matrix: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    """Return the determinant of an exact 3 by 3 matrix."""
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def determinant4(matrix: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    """Return the determinant of an exact 4 by 4 matrix."""
    result = Fraction(0)
    for column in range(4):
        minor = tuple(
            tuple(
                row[minor_column] for minor_column in range(4) if minor_column != column
            )
            for row in matrix[1:]
        )
        result += (
            (1 if column % 2 == 0 else -1) * matrix[0][column] * determinant3(minor)
        )
    return result
