"""Typed wire contracts for exact moments and orthogonal polynomials."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.math.moments_orthogonal.values import (
    MAX_MOMENTS,
    MAX_QUADRATURE_POINTS,
    MAX_RECURRENCE_ORDER,
    RecurrenceCoefficientsValue,
)

MAX_RATIONAL_DIGITS = 4_096

# Golub-Welsch converts admitted rationals to IEEE doubles; every accepted
# coefficient must convert to a finite double, beta_0 must survive conversion
# as a positive mass, and every subdiagonal entry must stay far from both
# overflow and underflow so its square root is exact enough.
MAX_QUADRATURE_MAGNITUDE = Fraction(10) ** 300
MIN_QUADRATURE_MAGNITUDE = Fraction(1, 10**300)


def _to_fractions(
    values: tuple[CanonicalRational, ...],
) -> tuple[Fraction, ...]:
    return tuple(v.as_fraction() for v in values)


def _from_fractions(
    values: tuple[Fraction, ...],
) -> tuple[CanonicalRational, ...]:
    return tuple(CanonicalRational.from_fraction(v) for v in values)


def _require_canonical_kernel_output(
    values: tuple[Fraction, ...],
    *,
    label: str,
) -> None:
    """Require every exact kernel output to fit the canonical rational limit.

    Per-component input caps cannot bound recurrence growth, so admission
    executes the bounded kernel and rejects any sequence whose complete
    exact output exceeds the canonical limit before the operation runs.
    """

    for value in values:
        try:
            CanonicalRational.from_fraction(value)
        except ValueError:
            raise ValueError(
                f"the exact {label} exceed the canonical "
                f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit limit for this input"
            ) from None


def _validate_moments(moments: tuple[CanonicalRational, ...]) -> None:
    if not 1 <= len(moments) <= MAX_MOMENTS:
        raise ValueError("moment sequence must contain between 1 and 64 moments")
    for value in moments:
        require_bounded_rational(value, max_digits=MAX_RATIONAL_DIGITS, label="moment")


def _validate_alpha_beta(
    alpha: tuple[CanonicalRational, ...],
    beta: tuple[CanonicalRational, ...],
) -> None:
    if not 1 <= len(beta) <= MAX_RECURRENCE_ORDER:
        raise ValueError("beta must contain between 1 and 16 entries")
    if not 0 <= len(alpha) <= MAX_RECURRENCE_ORDER:
        raise ValueError("alpha out of range")
    if len(alpha) != len(beta) and len(alpha) != len(beta) - 1:
        raise ValueError("alpha must have length len(beta)-1 or len(beta)")
    if beta[0].num == "0" or beta[0].num.startswith("-"):
        raise ValueError(
            "beta_0 (the zeroth moment of a positive functional) must be nonzero"
        )
    # beta_1, ..., beta_{n-1} are squared-norm ratios of a positive-definite
    # sequence and occupy the Jacobi subdiagonal; each must be positive.
    for index in range(1, min(len(alpha), len(beta))):
        if beta[index].num.startswith("-") or beta[index].num == "0":
            raise ValueError(
                "subdiagonal beta entries must be positive squared-norm ratios"
            )
    for value in (*alpha, *beta):
        require_bounded_rational(
            value, max_digits=MAX_RATIONAL_DIGITS, label="coefficient"
        )


# ---------------------------------------------------------------------------
# Hankel matrix
# ---------------------------------------------------------------------------


class HankelMatrixRequest(StrictModel):
    moments: tuple[CanonicalRational, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_moments(self) -> Self:
        _validate_moments(self.moments)
        return self


class HankelMatrixResult(HankelMatrixRequest):
    matrix: tuple[tuple[CanonicalRational, ...], ...]
    dimension: int = Field(ge=1)
    complete: Literal[True] = True
    method: Literal["EXACT_HANKEL_ASSEMBLY"] = "EXACT_HANKEL_ASSEMBLY"

    @model_validator(mode="after")
    def bind_hankel(self) -> Self:
        from jacobian.math.moments_orthogonal.operations import hankel_matrix

        result = hankel_matrix(_to_fractions(self.moments))
        if self.dimension != len(result.matrix):
            raise ValueError("dimension must match the Hankel matrix size")
        expected_matrix = tuple(
            tuple(CanonicalRational.from_fraction(v) for v in row)
            for row in result.matrix
        )
        if self.matrix != expected_matrix:
            raise ValueError("matrix must be the exact Hankel matrix")
        return self


# ---------------------------------------------------------------------------
# Recurrence coefficients
# ---------------------------------------------------------------------------


class RecurrenceCoefficientsRequest(StrictModel):
    moments: tuple[CanonicalRational, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_moments(self) -> Self:
        _validate_moments(self.moments)
        # The Gram-Schmidt kernel requires a positive-definite moment
        # functional; the zeroth moment is the squared norm of the constant
        # polynomial and must be positive even when the sequence is too short
        # to run the full recurrence.
        if self.moments[0].as_fraction() <= 0:
            raise ValueError("the zeroth moment must be nonzero")
        # Per-moment digit caps cannot bound exact recurrence-coefficient
        # growth, so admission replays the bounded kernel and requires its
        # complete output to fit the canonical limit.
        from jacobian.math.moments_orthogonal.operations import (
            recurrence_coefficients,
        )

        result = recurrence_coefficients(_to_fractions(self.moments))
        _require_canonical_kernel_output(
            (*result.alpha, *result.beta),
            label="recurrence coefficients",
        )
        return self


class RecurrenceCoefficientsResult(RecurrenceCoefficientsRequest):
    coefficients: RecurrenceCoefficientsValue
    complete: Literal[True] = True
    method: Literal["EXACT_GRAM_SCHMIDT"] = "EXACT_GRAM_SCHMIDT"

    @model_validator(mode="after")
    def bind_recurrence(self) -> Self:
        from jacobian.math.moments_orthogonal.operations import (
            recurrence_coefficients,
        )

        result = recurrence_coefficients(_to_fractions(self.moments))
        expected = RecurrenceCoefficientsValue(
            alpha=_from_fractions(result.alpha),
            beta=_from_fractions(result.beta),
        )
        if self.coefficients != expected:
            raise ValueError(
                "coefficients must be the exact recurrence coefficient value"
            )
        return self


# ---------------------------------------------------------------------------
# Jacobi matrix
# ---------------------------------------------------------------------------


class JacobiMatrixRequest(StrictModel):
    coefficients: RecurrenceCoefficientsValue

    @model_validator(mode="after")
    def require_valid_coefficients(self) -> Self:
        _validate_alpha_beta(self.coefficients.alpha, self.coefficients.beta)
        return self


class JacobiMatrixResult(JacobiMatrixRequest):
    diagonal: tuple[CanonicalRational, ...]
    off_diagonal: tuple[CanonicalRational, ...]
    complete: Literal[True] = True
    method: Literal["EXACT_TRIDIAGONAL_ASSEMBLY"] = "EXACT_TRIDIAGONAL_ASSEMBLY"

    @model_validator(mode="after")
    def bind_jacobi(self) -> Self:
        from jacobian.math.moments_orthogonal.operations import jacobi_matrix

        result = jacobi_matrix(
            _to_fractions(self.coefficients.alpha),
            _to_fractions(self.coefficients.beta),
        )
        if self.diagonal != _from_fractions(result.diagonal):
            raise ValueError("diagonal must match the exact Jacobi diagonal")
        if self.off_diagonal != _from_fractions(result.off_diagonal):
            raise ValueError("off_diagonal must match the exact Jacobi off-diagonal")
        return self


# ---------------------------------------------------------------------------
# Christoffel-Darboux kernel
# ---------------------------------------------------------------------------


class ChristoffelDarbouxRequest(StrictModel):
    coefficients: RecurrenceCoefficientsValue
    x: CanonicalRational
    y: CanonicalRational

    @model_validator(mode="after")
    def require_valid_coefficients(self) -> Self:
        _validate_alpha_beta(self.coefficients.alpha, self.coefficients.beta)
        # Bound x,y more tightly than the generic 4096: the recurrence
        # evaluates polynomials of degree up to len(alpha) and the kernel sums
        # n terms, so digit growth is roughly n * digit(x).  For n=16, even
        # 4095-digit x gives 122k-digit denominators.  Cap at 1024 to keep
        # worst-case outputs within the 32768-digit canonical limit.
        max_cd_digits = 1024
        require_bounded_rational(self.x, max_digits=max_cd_digits, label="x")
        require_bounded_rational(self.y, max_digits=max_cd_digits, label="y")
        # Coefficient numerators and denominators both drive exact kernel
        # growth, and no static per-component estimate bounds it, so
        # admission replays the bounded kernel and requires its complete
        # output to fit the canonical limit.
        from jacobian.math.moments_orthogonal.operations import (
            christoffel_darboux,
        )

        result = christoffel_darboux(
            _to_fractions(self.coefficients.alpha),
            _to_fractions(self.coefficients.beta),
            self.x.as_fraction(),
            self.y.as_fraction(),
        )
        _require_canonical_kernel_output(
            (result.kernel,),
            label="Christoffel-Darboux kernel",
        )
        _require_canonical_kernel_output(
            result.polynomials_evaluated,
            label="evaluated orthogonal polynomials",
        )
        return self


class ChristoffelDarbouxResult(ChristoffelDarbouxRequest):
    kernel: CanonicalRational
    polynomials_evaluated: tuple[CanonicalRational, ...]
    complete: Literal[True] = True
    method: Literal["EXACT_CD_RECURRENCE"] = "EXACT_CD_RECURRENCE"

    @model_validator(mode="after")
    def bind_christoffel_darboux(self) -> Self:
        from jacobian.math.moments_orthogonal.operations import (
            christoffel_darboux,
        )

        result = christoffel_darboux(
            _to_fractions(self.coefficients.alpha),
            _to_fractions(self.coefficients.beta),
            self.x.as_fraction(),
            self.y.as_fraction(),
        )
        if self.kernel != CanonicalRational.from_fraction(result.kernel):
            raise ValueError("kernel must be the exact Christoffel-Darboux kernel")
        if self.polynomials_evaluated != _from_fractions(result.polynomials_evaluated):
            raise ValueError(
                "polynomials_evaluated must match the evaluated polynomials"
            )
        return self


# ---------------------------------------------------------------------------
# Gaussian quadrature
# ---------------------------------------------------------------------------


class GaussianQuadratureRequest(StrictModel):
    coefficients: RecurrenceCoefficientsValue

    @model_validator(mode="after")
    def require_valid_coefficients(self) -> Self:
        alpha = self.coefficients.alpha
        beta = self.coefficients.beta
        if not 1 <= len(alpha) <= MAX_QUADRATURE_POINTS:
            raise ValueError("alpha must contain between 1 and 16 entries")
        if len(beta) != len(alpha) and len(beta) != len(alpha) + 1:
            raise ValueError("beta must have length len(alpha) or len(alpha)+1")
        beta_zero = beta[0].as_fraction()
        if beta_zero <= 0:
            raise ValueError(
                "beta_0 (the zeroth moment of a positive functional) must be nonzero"
            )
        if beta_zero < MIN_QUADRATURE_MAGNITUDE:
            raise ValueError(
                "beta_0 falls below the quadrature finite-double underflow bound"
            )
        # Subdiagonal entries feed math.sqrt after float conversion; they must
        # be positive and safely inside the finite IEEE-double range, and the
        # diagonal and mu_0 must convert to finite doubles without overflow.
        for index in range(1, min(len(alpha), len(beta))):
            sub = beta[index].as_fraction()
            if sub <= 0:
                raise ValueError(
                    "subdiagonal beta entries must be positive squared-norm ratios"
                )
            if sub < MIN_QUADRATURE_MAGNITUDE:
                raise ValueError(
                    "subdiagonal beta entries fall below the quadrature underflow bound"
                )
        for value in (*alpha, *beta):
            require_bounded_rational(
                value, max_digits=MAX_RATIONAL_DIGITS, label="coefficient"
            )
            if abs(value.as_fraction()) > MAX_QUADRATURE_MAGNITUDE:
                raise ValueError(
                    "quadrature coefficients exceed the finite-float magnitude bound"
                )
        return self


class GaussianQuadratureResult(GaussianQuadratureRequest):
    nodes: tuple[CanonicalRational, ...]
    weights: tuple[CanonicalRational, ...]
    complete: Literal[False] = False
    method: Literal["GOLUB_WELSCH_NUMERICAL"] = "GOLUB_WELSCH_NUMERICAL"
    approximation: Literal["IEEE_754_DOUBLE_DYADIC"] = "IEEE_754_DOUBLE_DYADIC"
    is_numerical_approximation: Literal[True] = True

    @model_validator(mode="after")
    def bind_gaussian_quadrature(self) -> Self:
        from jacobian.math.moments_orthogonal.operations import (
            gaussian_quadrature,
        )

        result = gaussian_quadrature(
            _to_fractions(self.coefficients.alpha),
            _to_fractions(self.coefficients.beta),
        )
        if self.nodes != _from_fractions(result.nodes):
            raise ValueError("nodes must match the Golub-Welsch eigenvalues")
        if self.weights != _from_fractions(result.weights):
            raise ValueError("weights must match the Golub-Welsch weights")
        return self


__all__ = [
    "ChristoffelDarbouxRequest",
    "ChristoffelDarbouxResult",
    "GaussianQuadratureRequest",
    "GaussianQuadratureResult",
    "HankelMatrixRequest",
    "HankelMatrixResult",
    "JacobiMatrixRequest",
    "JacobiMatrixResult",
    "RecurrenceCoefficientsRequest",
    "RecurrenceCoefficientsResult",
]
