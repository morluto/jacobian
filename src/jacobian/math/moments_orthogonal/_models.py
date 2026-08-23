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
)

MAX_RATIONAL_DIGITS = 4_096

# Golub-Welsch converts admitted rationals to IEEE doubles; every accepted
# coefficient must convert to a finite double and every subdiagonal entry must
# stay far from both overflow and underflow so its square root is exact enough.
MAX_QUADRATURE_MAGNITUDE = Fraction(10) ** 300
MIN_QUADRATURE_SUBDIAGONAL = Fraction(1, 10 ** 300)


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
        require_bounded_rational(
            value, max_digits=MAX_RATIONAL_DIGITS, label="moment"
        )


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

        derived = recurrence_coefficients(_to_fractions(self.moments))
        # Derived-coefficient growth budget: per-moment bounds do not bound
        # the exact Gram-Schmidt output, so admission must reject sequences
        # whose recurrence coefficients leave the canonical rational domain.
        try:
            _from_fractions(derived.alpha)
            _from_fractions(derived.beta)
        except ValueError as exc:
            raise ValueError(
                "derived recurrence coefficients exceed the canonical "
                "rational digit bound"
            ) from exc
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

        result = jacobi_matrix(
            _to_fractions(self.alpha), _to_fractions(self.beta)
        )
        if self.diagonal != _from_fractions(result.diagonal):
            raise ValueError("diagonal must match the exact Jacobi diagonal")
        if self.off_diagonal != _from_fractions(result.off_diagonal):
            raise ValueError(
                "off_diagonal must match the exact Jacobi off-diagonal"
            )
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
        require_bounded_rational(
            self.x, max_digits=MAX_RATIONAL_DIGITS, label="x"
        )
        require_bounded_rational(
            self.y, max_digits=MAX_RATIONAL_DIGITS, label="y"
        )
        # Degree-aware derived-growth budget: the forward recurrence
        # multiplies polynomials by (x - alpha_k) each step and adds
        # beta_k * p_{k-1}, so p_k carries at most
        # k*(digit(x)+1) + sum(alpha,beta digits) + k digits, and the kernel
        # sums n products of evaluated polynomials.  A plain sum of input
        # component digits would admit concentrated growth such as unit
        # coefficients with x = 10^3000, whose p_15 already exceeds the
        # canonical rational domain.
        def _digits(v: CanonicalRational) -> int:
            return max(len(v.num.lstrip("-")), len(v.den))

        alpha_digits = sum(_digits(v) for v in self.alpha)
        beta_digits = sum(_digits(v) for v in self.beta)
        x_digits = _digits(self.x)
        y_digits = _digits(self.y)
        order = len(self.alpha)
        slack = alpha_digits + beta_digits + order
        p_x_bound = order * (x_digits + 1) + slack
        p_y_bound = order * (y_digits + 1) + slack
        kernel_bound = order * (
            p_x_bound + p_y_bound + beta_digits + order + 2
        )
        if max(p_x_bound, p_y_bound, kernel_bound) > MAX_CANONICAL_RATIONAL_DIGITS:
            raise ValueError(
                "Christoffel-Darboux inputs can grow the exact kernel beyond "
                "the canonical rational digit bound"
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
            raise ValueError(
                "kernel must be the exact Christoffel-Darboux kernel"
            )
        if self.polynomials_evaluated != _from_fractions(
            result.polynomials_evaluated
        ):
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
        if (
            len(self.beta) != len(self.alpha)
            and len(self.beta) != len(self.alpha) + 1
        ):
            raise ValueError("beta must have length len(alpha) or len(alpha)+1")
        beta_zero = self.beta[0].as_fraction()
        if beta_zero <= 0:
            raise ValueError(
                "beta_0 (the zeroth moment of a positive functional) must be positive"
            )
        if beta_zero < MIN_QUADRATURE_SUBDIAGONAL:
            raise ValueError("beta_0 falls below the quadrature underflow bound")
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
    complete: Literal[False] = False
    method: Literal["GOLUB_WELSCH_APPROXIMATE"] = "GOLUB_WELSCH_APPROXIMATE"
    exactness: Literal["APPROXIMATE_DOUBLE"] = "APPROXIMATE_DOUBLE"

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
