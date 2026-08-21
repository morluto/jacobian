"""Typed wire contracts for exact moments and orthogonal polynomials."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.moments_orthogonal.values import (
    MAX_HANKEL_DIMENSION,
    MAX_MOMENTS,
    MAX_POLYNOMIAL_COUNT,
    MAX_QUADRATURE_POINTS,
    MAX_RECURRENCE_ORDER,
)

MAX_RATIONAL_DIGITS = 4_096


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
    if beta[0].num == "0":
        raise ValueError("beta_0 (the zeroth moment) must be nonzero")
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
        if self.beta[0].num == "0":
            raise ValueError("beta_0 (the zeroth moment) must be nonzero")
        for value in (*self.alpha, *self.beta):
            require_bounded_rational(
                value, max_digits=MAX_RATIONAL_DIGITS, label="coefficient"
            )
        return self


class GaussianQuadratureResult(GaussianQuadratureRequest):
    nodes: tuple[float, ...]
    weights: tuple[float, ...]
    complete: Literal[True] = True
    method: Literal["GOLUB_WELSCH"] = "GOLUB_WELSCH"

    @model_validator(mode="after")
    def bind_gaussian_quadrature(self) -> Self:
        from jacobian.math.moments_orthogonal.operations import (
            gaussian_quadrature,
        )

        result = gaussian_quadrature(
            _to_fractions(self.alpha), _to_fractions(self.beta)
        )
        if self.nodes != result.nodes:
            raise ValueError("nodes must match the Golub-Welsch eigenvalues")
        if self.weights != result.weights:
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
