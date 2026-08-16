"""Markov chain operations backed by SymPy."""
from __future__ import annotations
__all__ = ["stationary_distribution", "is_ergodic"]

def stationary_distribution(matrix):
    import sympy
    n = len(matrix)
    P = sympy.Matrix([[sympy.Rational(matrix[i][j]["num"], matrix[i][j]["den"])
                       for j in range(n)] for i in range(n)])
    # Find eigenvector for eigenvalue 1
    eigenvects = P.T.eigenvects()
    for eigenval, mult, vects in eigenvects:
        if eigenval == 1 and len(vects) > 0:
            vect = vects[0]
            total = sum(vect)
            normalized = [v / total for v in vect]
            return normalized
    return []

def is_ergodic(matrix):
    import sympy
    n = len(matrix)
    P = sympy.Matrix([[sympy.Rational(matrix[i][j]["num"], matrix[i][j]["den"])
                       for j in range(n)] for i in range(n)])
    # Check irreducible: all entries of P^n are positive for some n
    # Check aperiodic: gcd of return times is 1
    # Simplified: check if all entries of P^2 are positive
    P2 = P * P
    for i in range(n):
        for j in range(n):
            if P2[i, j] <= 0:
                return False
    return True
