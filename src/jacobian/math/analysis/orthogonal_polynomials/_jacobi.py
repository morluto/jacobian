"""Canonical-family kernel for finite Jacobi matrices."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.math.analysis.orthogonal_polynomials.values import (
    JacobiMatrix,
    OrthogonalPolynomialFamily,
    ThreeTermRecurrence,
)
from jacobian.math.matrices.values import rational_matrix_from_fractions


class JacobiMatrixAdmissionError(ValueError):
    """A native Jacobi-matrix admission failure with an owner-local code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class IncompatibleRecurrenceError(ValueError):
    """The supplied coefficients disprove the three-term relation."""


def _from_fraction(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(value)


def require_three_term_identities(
    family: OrthogonalPolynomialFamily,
    alphas: list[Fraction],
    betas: list[Fraction],
) -> None:
    """Establish the complete finite three-term identities of one family.

    For every ``k`` with ``p_{k+1}`` supplied, ``x*p_k - p_{k+1}`` must equal
    ``alpha_k*p_k + beta_k*p_{k-1}`` coefficient-wise (with ``p_{-1} = 0`` and
    ``beta_0 = 0``). Checking only the ``x^k`` coefficient (which defines
    ``alpha_k``) or only the norm ratios (which define ``beta_k``) accepts
    families whose remaining coefficients contradict the emitted recurrence.
    This finite prerequisite is shared by the recurrence and Jacobi consumers;
    callers map the ``ValueError`` to their owner-local admission error.
    """

    polys = [
        [coefficient.as_fraction() for coefficient in term.coefficients]
        for term in family.polynomials
    ]
    n = len(polys)
    if len(alphas) != max(n - 1, 0) or len(betas) < len(alphas):
        raise IncompatibleRecurrenceError(
            "recurrence coefficient dimensions do not match the family"
        )
    if n >= 1 and len(betas) > 0 and betas[0] != Fraction(0):
        raise IncompatibleRecurrenceError("beta[0] must be the unused zero placeholder")
    for k in range(len(alphas)):
        p_k = polys[k]
        p_next = polys[k + 1]
        x_pk = [Fraction(0)] * (len(p_k) + 1)
        for i, coefficient in enumerate(p_k):
            x_pk[i + 1] = coefficient
        width = max(len(x_pk), len(p_next))
        lhs = [
            (x_pk[i] if i < len(x_pk) else Fraction(0))
            - (p_next[i] if i < len(p_next) else Fraction(0))
            for i in range(width)
        ]
        alpha_k = alphas[k]
        beta_k = betas[k]
        rhs = [Fraction(0)] * width
        for i, coefficient in enumerate(p_k):
            if i < width:
                rhs[i] += alpha_k * coefficient
        if k > 0:
            p_prev = polys[k - 1]
            for i, coefficient in enumerate(p_prev):
                if i < width:
                    rhs[i] += beta_k * coefficient
        if lhs != rhs:
            raise IncompatibleRecurrenceError(
                f"family polynomials contradict the three-term recurrence at k={k}: "
                "x*p_k - p_{k+1} differs from alpha_k*p_k + beta_k*p_{k-1}"
            )


def _derive_jacobi_coefficients(
    family: OrthogonalPolynomialFamily,
) -> tuple[list[Fraction], list[Fraction]]:
    """Derive the exact alpha/beta lists emitted by the Jacobi kernel."""

    polys = family.polynomials
    n = len(polys)
    alphas: list[Fraction] = []
    betas: list[Fraction] = [Fraction(0)]
    for k in range(n):
        p_k = [coefficient.as_fraction() for coefficient in polys[k].coefficients]
        if k > 0:
            betas.append(
                polys[k].squared_norm.as_fraction()
                / polys[k - 1].squared_norm.as_fraction()
            )
        if k + 1 >= n:
            continue
        p_next = [
            coefficient.as_fraction() for coefficient in polys[k + 1].coefficients
        ]
        if k == 0:
            alphas.append(-p_next[0])
            continue
        x_pk = [Fraction(0)] * (len(p_k) + 1)
        for i, coefficient in enumerate(p_k):
            x_pk[i + 1] = coefficient
        residual = [
            x_pk[i] - p_next[i] if i < len(p_next) else x_pk[i]
            for i in range(len(x_pk))
        ]
        alphas.append(residual[k] if k < len(residual) else Fraction(0))
    return alphas, betas


def require_jacobi_matrix_admission(
    family: OrthogonalPolynomialFamily,
) -> tuple[list[Fraction], list[Fraction]]:
    """Establish applicability once and retain the bounded recurrence entries."""

    polys = family.polynomials
    for k in range(1, len(polys)):
        if polys[k - 1].squared_norm.num == 0:
            raise JacobiMatrixAdmissionError(
                "norm_ratio",
                f"adjacent-norm ratio beta_{k} is undefined because squared "
                f"norm h_{k - 1} vanishes; supply a "
                "family with nonzero norms for every emitted ratio",
            )
    alphas, betas = _derive_jacobi_coefficients(family)
    digit_limit = 10**MAX_CANONICAL_RATIONAL_DIGITS
    for k, value in enumerate(alphas):
        if abs(value.numerator) >= digit_limit or value.denominator >= digit_limit:
            raise JacobiMatrixAdmissionError(
                "recurrence_height",
                f"derived recurrence entry alpha_{k} exceeds the canonical "
                "rational digit limit; supply a family whose coefficient "
                "differences stay representable",
            )
    for k, value in enumerate(betas[1:], start=1):
        if abs(value.numerator) >= digit_limit or value.denominator >= digit_limit:
            raise JacobiMatrixAdmissionError(
                "norm_ratio_height",
                f"adjacent-norm ratio beta_{k} exceeds the canonical rational "
                "digit limit; supply a family whose squared norm ratios stay "
                "representable",
            )
    try:
        require_three_term_identities(family, alphas, betas)
    except IncompatibleRecurrenceError as exc:
        raise JacobiMatrixAdmissionError("incompatible_family", str(exc)) from None
    return alphas, betas


def jacobi_matrix_from_family(
    family: OrthogonalPolynomialFamily,
    coefficients: tuple[list[Fraction], list[Fraction]],
) -> JacobiMatrix:
    """Compute the exact finite Jacobi matrix of one admitted family."""
    polys = family.polynomials
    n = len(polys)

    if n < 2:
        recurrence = ThreeTermRecurrence._from_kernel(
            alpha=(),
            beta=(CanonicalRational.from_integer_ratio(0, 1),),
            variable=family.variable,
        )
        return JacobiMatrix._from_kernel(
            family=family,
            recurrence=recurrence,
            matrix=rational_matrix_from_fractions((), column_count=0),
        )

    alphas, betas = coefficients

    matrix_size = n - 1
    matrix = [[Fraction(0)] * matrix_size for _ in range(matrix_size)]
    for i in range(matrix_size):
        matrix[i][i] = alphas[i]
        if i < matrix_size - 1:
            matrix[i + 1][i] = Fraction(1)
            matrix[i][i + 1] = betas[i + 1]

    recurrence = ThreeTermRecurrence._from_kernel(
        alpha=tuple(_from_fraction(alpha) for alpha in alphas),
        beta=tuple(_from_fraction(beta) for beta in betas),
        variable=family.variable,
    )
    return JacobiMatrix._from_kernel(
        family=family,
        recurrence=recurrence,
        matrix=rational_matrix_from_fractions(
            tuple(
                tuple(matrix[i][j] for j in range(matrix_size))
                for i in range(matrix_size)
            ),
            column_count=matrix_size,
        ),
    )
