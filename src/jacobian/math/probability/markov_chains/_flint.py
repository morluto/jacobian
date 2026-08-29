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


__all__ = ["solve_stationary_class"]
