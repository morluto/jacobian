"""Canonical-family kernel for finite Jacobi matrices."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.math.analysis.orthogonal_polynomials.values import (
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
        raise ValueError("recurrence coefficient dimensions do not match the family")
    if n >= 1 and len(betas) > 0 and betas[0] != Fraction(0):
        raise ValueError("beta[0] must be the unused zero placeholder")
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
            raise ValueError(
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
    betas: list[Fraction] = []
    for k in range(n - 1):
        p_k = [coefficient.as_fraction() for coefficient in polys[k].coefficients]
        p_next = [
            coefficient.as_fraction() for coefficient in polys[k + 1].coefficients
        ]
        if k == 0:
            alphas.append(-p_next[0])
            betas.append(Fraction(0))
        else:
            x_pk = [Fraction(0)] * (len(p_k) + 1)
            for i, coefficient in enumerate(p_k):
                x_pk[i + 1] = coefficient
            residual = [
                x_pk[i] - p_next[i] if i < len(p_next) else x_pk[i]
                for i in range(len(x_pk))
            ]
            alphas.append(residual[k] if k < len(residual) else Fraction(0))
            betas.append(
                polys[k].squared_norm.as_fraction()
                / polys[k - 1].squared_norm.as_fraction()
            )
    return alphas, betas


def _require_compatible_or_raise(family: OrthogonalPolynomialFamily) -> None:
    """Reject families whose coefficients contradict their norm ratios."""

    if len(family.polynomials) < 2:
        return
    try:
        alphas, betas = _derive_jacobi_coefficients(family)
        require_three_term_identities(family, alphas, betas)
    except ValueError as exc:
        # Preserve zero-division diagnostics raised for vanishing norms;
        # only compatibility failures are mapped here (admission already
        # rejects undefined ratios above).
        if "contradict" not in str(exc) and "dimensions" not in str(exc):
            raise
        raise JacobiMatrixAdmissionError(
            "incompatible_family",
            str(exc),
        ) from None


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
    _require_compatible_or_raise(family)


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
            try:
                betas.append(squared_norm_k / squared_norm_prev)
            except ZeroDivisionError:
                raise JacobiMatrixAdmissionError(
                    "norm_ratio",
                    f"adjacent-norm ratio beta_{k} is undefined because squared "
                    "norm h_{k-1} vanishes",
                ) from None
    try:
        require_three_term_identities(family, alphas, betas)
    except ValueError as exc:
        raise JacobiMatrixAdmissionError(
            "incompatible_family",
            str(exc),
        ) from None

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
