"""Canonical-family kernel for finite Jacobi matrices."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.math.moments_orthogonal.values import (
    JacobiMatrix,
    OrthogonalPolynomialFamily,
)


class JacobiMatrixAdmissionError(ValueError):
    """A native Jacobi-matrix admission failure with an owner-local code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _from_fraction(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(value)


def require_jacobi_matrix_admission(family: OrthogonalPolynomialFamily) -> None:
    """Bound every exact entry emitted by the Jacobi-matrix kernel.

    The canonical family is intentionally accepted directly: this is the
    owner-local mathematical admission shared by the MCP request validator and
    the native API, not a request-envelope reconstruction.
    """
    polys = family.polynomials
    digit_limit = 10**MAX_CANONICAL_RATIONAL_DIGITS
    for k in range(len(polys) - 1):
        p_k = [coefficient.as_fraction() for coefficient in polys[k].coefficients]
        p_next = [
            coefficient.as_fraction() for coefficient in polys[k + 1].coefficients
        ]
        if k == 0:
            alpha_k = -p_next[0]
        else:
            x_pk = [Fraction(0)] * (len(p_k) + 1)
            for i, coefficient in enumerate(p_k):
                x_pk[i + 1] = coefficient
            residual = [
                x_pk[i] - p_next[i] if i < len(p_next) else x_pk[i]
                for i in range(len(x_pk))
            ]
            alpha_k = residual[k] if k < len(residual) else Fraction(0)
        if abs(alpha_k.numerator) >= digit_limit or alpha_k.denominator >= digit_limit:
            raise JacobiMatrixAdmissionError(
                "recurrence_height",
                f"derived recurrence entry alpha_{k} exceeds the canonical "
                "rational digit limit; supply a family whose coefficient "
                "differences stay representable",
            )
    for k in range(1, len(polys) - 1):
        h_k = polys[k].squared_norm.as_fraction()
        h_prev = polys[k - 1].squared_norm.as_fraction()
        if h_prev == 0 or h_k == 0:
            raise JacobiMatrixAdmissionError(
                "norm_ratio",
                f"adjacent-norm ratio beta_{k} is undefined because squared "
                f"norm h_{k - 1 if h_prev == 0 else k} vanishes; supply a "
                "family with nonzero norms for every emitted ratio",
            )
        ratio = h_k / h_prev
        if abs(ratio.numerator) >= digit_limit or ratio.denominator >= digit_limit:
            raise JacobiMatrixAdmissionError(
                "norm_ratio_height",
                f"adjacent-norm ratio beta_{k} exceeds the canonical rational "
                "digit limit; supply a family whose squared norm ratios stay "
                "representable",
            )


def jacobi_matrix_from_family(family: OrthogonalPolynomialFamily) -> JacobiMatrix:
    """Compute the exact finite Jacobi matrix of one admitted family."""
    polys = family.polynomials
    n = len(polys)

    if n < 2:
        return JacobiMatrix._from_kernel(
            alphas=(),
            betas=(),
            matrix=(),
            variable=family.variable,
        )

    alphas: list[Fraction] = []
    betas: list[Fraction] = []
    for k in range(n - 1):
        p_k = [coefficient.as_fraction() for coefficient in polys[k].coefficients]
        p_next = [
            coefficient.as_fraction() for coefficient in polys[k + 1].coefficients
        ]
        squared_norm_k = polys[k].squared_norm.as_fraction()

        if k == 0:
            alphas.append(-p_next[0])
            betas.append(Fraction(0))
        else:
            squared_norm_prev = polys[k - 1].squared_norm.as_fraction()
            x_pk = [Fraction(0)] * (len(p_k) + 1)
            for i, coefficient in enumerate(p_k):
                x_pk[i + 1] = coefficient
            residual = [
                x_pk[i] - p_next[i] if i < len(p_next) else x_pk[i]
                for i in range(len(x_pk))
            ]
            alphas.append(residual[k] if k < len(residual) else Fraction(0))
            betas.append(squared_norm_k / squared_norm_prev)

    matrix_size = n - 1
    matrix = [[Fraction(0)] * matrix_size for _ in range(matrix_size)]
    for i in range(matrix_size):
        matrix[i][i] = alphas[i]
        if i < matrix_size - 1:
            matrix[i + 1][i] = Fraction(1)
            matrix[i][i + 1] = betas[i + 1]

    return JacobiMatrix._from_kernel(
        alphas=tuple(_from_fraction(alpha) for alpha in alphas),
        betas=tuple(_from_fraction(beta) for beta in betas),
        matrix=tuple(
            tuple(_from_fraction(matrix[i][j]) for j in range(matrix_size))
            for i in range(matrix_size)
        ),
        variable=family.variable,
    )
