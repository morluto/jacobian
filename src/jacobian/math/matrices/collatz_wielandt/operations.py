"""Collatz-Wielandt quotient profile kernel."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.matrices.collatz_wielandt._models import (
    CollatzWielandtResult,
)

__all__ = ["compute_collatz_wielandt_profile"]


def compute_collatz_wielandt_profile(
    matrix: tuple[tuple[CanonicalRational, ...], ...],
    vector: tuple[CanonicalRational, ...],
) -> CollatzWielandtResult:
    """Return the componentwise quotient profile (Ax)_i / x_i."""
    n = len(vector)
    mat = [[matrix[i][j].as_fraction() for j in range(n)] for i in range(n)]
    vec = [v.as_fraction() for v in vector]

    quotients: list[Fraction] = []
    for i in range(n):
        ax_i = sum(mat[i][j] * vec[j] for j in range(n))
        if vec[i] == 0:
            raise ValueError("vector entries must be strictly positive")
        quotients.append(ax_i / vec[i])

    max_q = max(quotients)

    return CollatzWielandtResult(
        matrix=matrix,
        vector=vector,
        quotients=tuple(CanonicalRational.from_fraction(q) for q in quotients),
        max_quotient=CanonicalRational.from_fraction(max_q),
    )
