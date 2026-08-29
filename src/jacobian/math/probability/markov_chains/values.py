"""Canonical exact values used by Markov-chain operations."""

from __future__ import annotations

from fractions import Fraction
from math import factorial

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational

MAX_TRANSITION_STATES = 32
MAX_STATIONARY_STATES = 128
MAX_STATIONARY_SOLVE_WORK = 1_000_000

type TransitionMatrix = tuple[tuple[Fraction, ...], ...]


class TransitionMatrixAdmissionError(ValueError):
    """A semantic transition-matrix rejection with a stable reason and path."""

    def __init__(
        self,
        reason: str,
        message: str,
        location: tuple[str | int, ...] = ("matrix",),
    ) -> None:
        self.reason = reason
        self.location = location
        super().__init__(message)


def as_transition_matrix(
    matrix: tuple[tuple[CanonicalRational, ...], ...],
) -> TransitionMatrix:
    """Project parsed canonical rationals to the exact native matrix value."""

    return tuple(tuple(value.as_fraction() for value in row) for row in matrix)


def as_canonical_transition_matrix(
    matrix: TransitionMatrix,
) -> tuple[tuple[CanonicalRational, ...], ...]:
    """Project a native exact matrix into its canonical serialized value."""

    return tuple(
        tuple(CanonicalRational.from_fraction(value) for value in row) for row in matrix
    )


def require_transition_matrix(
    matrix: TransitionMatrix, *, maximum_states: int = MAX_TRANSITION_STATES
) -> None:
    """Admit one finite exact stochastic matrix before a native kernel runs."""

    dimension = len(matrix)
    if not 1 <= dimension <= maximum_states:
        raise TransitionMatrixAdmissionError(
            "transition_matrix_dimension",
            f"transition matrix dimension must be between 1 and {maximum_states}",
        )
    if any(len(row) != dimension for row in matrix):
        raise TransitionMatrixAdmissionError(
            "transition_matrix_not_square",
            "transition matrix must be square",
        )
    for row_index, row in enumerate(matrix):
        if any(value < 0 for value in row):
            raise TransitionMatrixAdmissionError(
                "transition_probability_negative",
                "transition probabilities must be nonnegative",
                ("matrix", row_index),
            )
        if sum(row) != 1:
            raise TransitionMatrixAdmissionError(
                "transition_row_not_stochastic",
                "each transition row must sum to one",
                ("matrix", row_index),
            )


def _decimal_digits(value: int) -> int:
    """Return decimal digits without serializing a potentially huge integer."""

    value = abs(value)
    if value == 0:
        return 1
    estimate = value.bit_length() * 30_103 // 100_000 + 1
    while value < 10 ** (estimate - 1):
        estimate -= 1
    return estimate


def require_stationary_distribution_admission(matrix: TransitionMatrix) -> None:
    """Bound exact stationary coordinates for a native transition matrix."""

    require_transition_matrix(matrix, maximum_states=MAX_STATIONARY_STATES)
    dimension = len(matrix)
    if dimension**3 > MAX_STATIONARY_SOLVE_WORK:
        raise TransitionMatrixAdmissionError(
            "stationary_solve_work_exceeds_bound",
            "stationary closed-class systems exceed the exact solve-work bound",
        )
    row_bounds: list[int] = []
    for column in range(dimension - 1):
        entries = tuple(matrix[row][column] for row in range(dimension))
        denominator_digits = sum(
            _decimal_digits(value.denominator) for value in entries
        )
        row_bounds.append(
            max(
                max(
                    _decimal_digits(value.numerator), _decimal_digits(value.denominator)
                )
                + 1
                + denominator_digits
                - _decimal_digits(value.denominator)
                for value in entries
            )
        )
    row_bounds.append(1)
    determinant_digits = sum(row_bounds) + len(str(factorial(dimension)))
    if determinant_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise TransitionMatrixAdmissionError(
            "stationary_height_exceeds_bound",
            "stationary distribution rational height exceeds the "
            f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit result bound",
        )


__all__ = ["TransitionMatrix"]
