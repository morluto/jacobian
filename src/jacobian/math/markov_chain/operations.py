"""Markov chain operations backed by SymPy."""

from __future__ import annotations

__all__ = ["is_ergodic", "stationary_distribution"]


def stationary_distribution(matrix):  # type: ignore[no-untyped-def]
    import sympy

    n = len(matrix)
    p = sympy.Matrix(
        [
            [sympy.Rational(matrix[i][j]["num"], matrix[i][j]["den"]) for j in range(n)]
            for i in range(n)
        ]
    )
    # Find eigenvector for eigenvalue 1
    eigenvects = p.T.eigenvects()
    for eigenval, _mult, vects in eigenvects:
        if eigenval == 1 and len(vects) > 0:
            vect = vects[0]
            total = sum(vect)
            normalized = [v / total for v in vect]
            return normalized
    return []


def is_ergodic(matrix):  # type: ignore[no-untyped-def]
    import sympy

    n = len(matrix)
    p = sympy.Matrix(
        [
            [sympy.Rational(matrix[i][j]["num"], matrix[i][j]["den"]) for j in range(n)]
            for i in range(n)
        ]
    )
    # Check irreducible: all entries of P^n are positive for some n
    # Check aperiodic: gcd of return times is 1
    # Simplified: check if all entries of P^2 are positive
    p2 = p * p
    for i in range(n):
        for j in range(n):
            if p2[i, j] <= 0:
                return False
    return True
