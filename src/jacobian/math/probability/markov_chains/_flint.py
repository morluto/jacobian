"""Private FLINT adapters for exact finite Markov chains."""

from fractions import Fraction

from jacobian.math.probability.markov_chains.values import TransitionMatrix


def solve_stationary_class(
    matrix: TransitionMatrix, closed_class: tuple[int, ...]
) -> tuple[Fraction, ...]:
    """Solve the normalized stationary system on one closed class."""

    from flint import fmpq, fmpq_mat

    size = len(closed_class)
    equations = [
        [
            fmpq(
                matrix[closed_class[column]][closed_class[row]].numerator,
                matrix[closed_class[column]][closed_class[row]].denominator,
            )
            - int(row == column)
            for column in range(size)
        ]
        for row in range(size)
    ]
    equations[-1] = [fmpq(1) for _ in range(size)]
    rhs = fmpq_mat([[0] for _ in range(size)])
    rhs[size - 1, 0] = 1
    solution = fmpq_mat(equations).solve(rhs)
    return tuple(
        Fraction(
            int(solution[index, 0].numerator),
            int(solution[index, 0].denominator),
        )
        for index in range(size)
    )


def solve_linear_system(
    a: list[list[Fraction]], b: list[Fraction]
) -> list[Fraction] | None:
    """Solve *Ax = b* exactly via FLINT's ``fmpq_mat.solve``.

    Returns the solution vector, or ``None`` when the system is singular.
    """

    from flint import fmpq, fmpq_mat

    n = len(a)
    if n == 0:
        return []

    coefficients = fmpq_mat(
        [[fmpq(val.numerator, val.denominator) for val in row] for row in a]
    )
    rhs = fmpq_mat([[fmpq(val.numerator, val.denominator)] for val in b])
    try:
        solution = coefficients.solve(rhs)
    except (ZeroDivisionError, ValueError, ArithmeticError):
        return None

    return [
        Fraction(
            int(solution[i, 0].numerator),
            int(solution[i, 0].denominator),
        )
        for i in range(n)
    ]


__all__ = ["solve_stationary_class", "solve_linear_system"]
