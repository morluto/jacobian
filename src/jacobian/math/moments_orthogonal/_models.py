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
from jacobian.math._rational_height import RationalHeight
from jacobian.math.moments_orthogonal.values import (
    MAX_MOMENTS,
    MAX_QUADRATURE_POINTS,
    MAX_RECURRENCE_ORDER,
)

MAX_RATIONAL_DIGITS = 4_096

# Golub-Welsch converts admitted rationals to IEEE doubles; every accepted
# coefficient must convert to a finite double and every subdiagonal entry must
# stay far from both overflow and underflow so its square root is exact enough.
MAX_QUADRATURE_MAGNITUDE = Fraction(10) ** 300
MIN_QUADRATURE_SUBDIAGONAL = Fraction(1, 10**300)


def _to_fractions(
    values: tuple[CanonicalRational, ...],
) -> tuple[Fraction, ...]:
    return tuple(v.as_fraction() for v in values)


def _from_fractions(values) -> tuple[CanonicalRational, ...]:
    return tuple(CanonicalRational.from_fraction(v) for v in values)


def _validate_moments(moments: tuple[CanonicalRational, ...]) -> None:
    if not 1 <= len(moments) <= MAX_MOMENTS:
        raise ValueError("moment sequence must contain between 1 and 64 moments")
    for value in moments:
        require_bounded_rational(value, max_digits=MAX_RATIONAL_DIGITS, label="moment")


def _coefficient_height(value: CanonicalRational) -> int:
    height = RationalHeight.from_canonical(value)
    return max(height.numerator_digits, height.denominator_digits)


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
            "beta_0 (the zeroth moment of a positive functional) must be positive"
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
        # functional; admit exactly the sequences it accepts so an accepted
        # request cannot fail inside execution.
        from jacobian.math.moments_orthogonal.operations import (
            recurrence_coefficients,
        )

        recurrence_coefficients(_to_fractions(self.moments))
        return self


class RecurrenceCoefficientsResult(RecurrenceCoefficientsRequest):
    alpha: tuple[CanonicalRational, ...]
    beta: tuple[CanonicalRational, ...]
    complete: Literal[True] = True
    method: Literal["EXACT_GRAM_SCHMIDT"] = "EXACT_GRAM_SCHMIDT"

    @model_validator(mode="after")
    def bind_recurrence(self) -> Self:
        from jacobian.math.moments_orthogonal.operations import (
            recurrence_coefficients,
        )

        result = recurrence_coefficients(_to_fractions(self.moments))
        if self.alpha != _from_fractions(result.alpha):
            raise ValueError("alpha must be the exact recurrence coefficients")
        if self.beta != _from_fractions(result.beta):
            raise ValueError("beta must be the exact recurrence coefficients")
        return self


# ---------------------------------------------------------------------------
# Jacobi matrix
# ---------------------------------------------------------------------------


class JacobiMatrixRequest(StrictModel):
    alpha: tuple[CanonicalRational, ...] = Field(min_length=0)
    beta: tuple[CanonicalRational, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_coefficients(self) -> Self:
        _validate_alpha_beta(self.alpha, self.beta)
        return self


class JacobiMatrixResult(JacobiMatrixRequest):
    diagonal: tuple[CanonicalRational, ...]
    off_diagonal: tuple[CanonicalRational, ...]
    complete: Literal[True] = True
    method: Literal["EXACT_TRIDIAGONAL_ASSEMBLY"] = "EXACT_TRIDIAGONAL_ASSEMBLY"

    @model_validator(mode="after")
    def bind_jacobi(self) -> Self:
        from jacobian.math.moments_orthogonal.operations import jacobi_matrix

        result = jacobi_matrix(_to_fractions(self.alpha), _to_fractions(self.beta))
        if self.diagonal != _from_fractions(result.diagonal):
            raise ValueError("diagonal must match the exact Jacobi diagonal")
        if self.off_diagonal != _from_fractions(result.off_diagonal):
            raise ValueError("off_diagonal must match the exact Jacobi off-diagonal")
        return self


# ---------------------------------------------------------------------------
# Christoffel-Darboux kernel
# ---------------------------------------------------------------------------


class ChristoffelDarbouxRequest(StrictModel):
    alpha: tuple[CanonicalRational, ...] = Field(min_length=0)
    beta: tuple[CanonicalRational, ...] = Field(min_length=1)
    x: CanonicalRational
    y: CanonicalRational

    @model_validator(mode="after")
    def require_valid_coefficients(self) -> Self:
        _validate_alpha_beta(self.alpha, self.beta)
        require_bounded_rational(self.x, max_digits=MAX_RATIONAL_DIGITS, label="x")
        require_bounded_rational(self.y, max_digits=MAX_RATIONAL_DIGITS, label="y")
        # Christoffel-Darboux output growth: each monic recurrence step
        # p_{k+1}(t) = (t - alpha_k) p_k(t) - beta_k p_{k-1}(t) raises the
        # evaluated height by at most H = height(x/y) + height(alpha)
        # + height(beta) + 2 digits, so p_k(t) has height at most k*H and,
        # dividing by h_k = prod(beta), every kernel summand has height at
        # most 2*k*H + (k+1)*height(beta). Summing all summands bounds the
        # exact kernel; reject any admitted request whose worst case could
        # exceed the canonical rational digit limit during execution.
        point_height = max(_coefficient_height(self.x), _coefficient_height(self.y))
        alpha_height = max((_coefficient_height(v) for v in self.alpha), default=1)
        beta_height = max((_coefficient_height(v) for v in self.beta), default=1)
        step = point_height + alpha_height + beta_height + 2
        kernel_bound = sum(
            2 * k * step + (k + 1) * beta_height + 2 for k in range(len(self.alpha))
        )
        if (
            kernel_bound + len(str(max(len(self.alpha), 1)))
            > MAX_CANONICAL_RATIONAL_DIGITS
        ):
            raise ValueError(
                "Christoffel-Darboux coefficients exceed the conservative "
                "combined order-and-height bound for an exact kernel"
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
            _to_fractions(self.alpha),
            _to_fractions(self.beta),
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
    alpha: tuple[CanonicalRational, ...] = Field(min_length=1)
    beta: tuple[CanonicalRational, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_coefficients(self) -> Self:
        if not 1 <= len(self.alpha) <= MAX_QUADRATURE_POINTS:
            raise ValueError("alpha must contain between 1 and 16 entries")
        if len(self.beta) != len(self.alpha) and len(self.beta) != len(self.alpha) + 1:
            raise ValueError("beta must have length len(alpha) or len(alpha)+1")
        beta_zero = self.beta[0].as_fraction()
        if beta_zero <= 0:
            raise ValueError(
                "beta_0 (the zeroth moment of a positive functional) must be positive"
            )
        # Subdiagonal entries feed math.sqrt after float conversion; they must
        # be positive and safely inside the finite IEEE-double range, and the
        # diagonal and mu_0 must convert to finite doubles without overflow.
        for index in range(1, min(len(self.alpha), len(self.beta))):
            sub = self.beta[index].as_fraction()
            if sub <= 0:
                raise ValueError(
                    "subdiagonal beta entries must be positive squared-norm ratios"
                )
            if sub < MIN_QUADRATURE_SUBDIAGONAL:
                raise ValueError(
                    "subdiagonal beta entries fall below the quadrature underflow bound"
                )
        for value in (*self.alpha, *self.beta):
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
    complete: Literal[True] = True
    method: Literal["GOLUB_WELSCH_FLOAT64_APPROX"] = "GOLUB_WELSCH_FLOAT64_APPROX"
    is_approximate: Literal[True] = True
    precision: Literal["FLOAT64"] = "FLOAT64"
    exactness: Literal["APPROXIMATE_DYADIC"] = "APPROXIMATE_DYADIC"

    @model_validator(mode="after")
    def bind_gaussian_quadrature(self) -> Self:
        from jacobian.math.moments_orthogonal.operations import (
            gaussian_quadrature,
        )

        result = gaussian_quadrature(
            _to_fractions(self.alpha), _to_fractions(self.beta)
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
