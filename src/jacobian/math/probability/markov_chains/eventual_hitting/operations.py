"""Eventual hitting probability kernel."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.probability.markov_chains.eventual_hitting._models import (
    EventualHittingProfileResult,
)

__all__ = ["compute_eventual_hitting_profile"]


def compute_eventual_hitting_profile(
    matrix: tuple[tuple[CanonicalRational, ...], ...],
    target_states: tuple[int, ...],
) -> EventualHittingProfileResult:
    """Return the eventual hitting probability profile for a Markov chain.

    For each state i, compute h(i) = P_i(ever hit the target set A).
    """
    n = len(matrix)
    target_set = set(target_states)
    m = [[matrix[i][j].as_fraction() for j in range(n)] for i in range(n)]
    h = [Fraction(0)] * n
    for i in target_set:
        h[i] = Fraction(1)

    non_target = [i for i in range(n) if i not in target_set]

    if not non_target:
        almost_sure = tuple(i for i in range(n))
        return EventualHittingProfileResult(
            matrix=matrix,
            target_states=target_states,
            hitting_probabilities=tuple(
                CanonicalRational.from_fraction(h[i]) for i in range(n)
            ),
            zero_states=(),
            proper_states=(),
            almost_sure_states=almost_sure,
        )

    a_matrix = [[Fraction(0)] * len(non_target) for _ in range(len(non_target))]
    b_vector = [Fraction(0)] * len(non_target)
    for row_idx, i in enumerate(non_target):
        a_matrix[row_idx][row_idx] = Fraction(1)
        for col_idx, j in enumerate(non_target):
            a_matrix[row_idx][col_idx] -= m[i][j]
        for j in target_set:
            b_vector[row_idx] += m[i][j] * Fraction(1)

    solution = _solve_linear_system(a_matrix, b_vector)
    for idx, i in enumerate(non_target):
        h[i] = solution[idx] if solution is not None else Fraction(0)

    zero_states = tuple(i for i in range(n) if h[i] == 0)
    proper_states = tuple(i for i in range(n) if 0 < h[i] < 1)
    almost_sure = tuple(i for i in range(n) if h[i] == 1)

    return EventualHittingProfileResult(
        matrix=matrix,
        target_states=target_states,
        hitting_probabilities=tuple(
            CanonicalRational.from_fraction(h[i]) for i in range(n)
        ),
        zero_states=zero_states,
        proper_states=proper_states,
        almost_sure_states=almost_sure,
    )


def _solve_linear_system(
    a: list[list[Fraction]], b: list[Fraction]
) -> list[Fraction] | None:
    """Solve Ax = b using Gaussian elimination with exact fractions."""
    n = len(a)
    aug = [[*list(a[i]), b[i]] for i in range(n)]

    for col in range(n):
        pivot = None
        for row in range(col, n):
            if aug[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(n):
            if row != col and aug[row][col] != 0:
                factor = aug[row][col] / aug[col][col]
                for k in range(n + 1):
                    aug[row][k] -= factor * aug[col][k]

    solution = [Fraction(0)] * n
    for i in range(n):
        if aug[i][i] != 0:
            solution[i] = aug[i][n] / aug[i][i]
    return solution
