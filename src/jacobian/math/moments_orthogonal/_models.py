"""Typed wire contracts for exact moments and orthogonal polynomials."""

from __future__ import annotations

from collections.abc import Iterable
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
from jacobian.math.matrices.values import RationalMatrix
from jacobian.math.moments_orthogonal.values import (
    MAX_MOMENTS,
    MAX_RATIONAL_DIGITS,
    RecurrenceCoefficients,
)

# Golub-Welsch converts admitted rationals to IEEE doubles; every accepted
# coefficient must convert to a finite double and every subdiagonal entry must
# stay far from both overflow and underflow so its square root is exact enough.
MAX_QUADRATURE_MAGNITUDE = Fraction(10) ** 300
MIN_QUADRATURE_SUBDIAGONAL = Fraction(1, 10**300)


def _to_fractions(
    values: tuple[CanonicalRational, ...],
) -> tuple[Fraction, ...]:
    return tuple(v.as_fraction() for v in values)


def _from_fractions(values: Iterable[Fraction]) -> tuple[CanonicalRational, ...]:
    return tuple(CanonicalRational.from_fraction(v) for v in values)


def _validate_moments(moments: tuple[CanonicalRational, ...]) -> None:
    if not 1 <= len(moments) <= MAX_MOMENTS:
        raise ValueError("moment sequence must contain between 1 and 64 moments")
    for value in moments:
        require_bounded_rational(value, max_digits=MAX_RATIONAL_DIGITS, label="moment")


def _coefficient_height(value: CanonicalRational) -> int:
    height = RationalHeight.from_canonical(value)
    return max(height.numerator_digits, height.denominator_digits)


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
    matrix: RationalMatrix
    dimension: int = Field(ge=1)
    complete: Literal[True] = True
    method: Literal["EXACT_HANKEL_ASSEMBLY"] = "EXACT_HANKEL_ASSEMBLY"

    @model_validator(mode="after")
    def bind_hankel(self) -> Self:
        from jacobian.math.moments_orthogonal.operations import hankel_matrix

        result = hankel_matrix(_to_fractions(self.moments))
        expected_matrix = RationalMatrix(
            entries=tuple(
                tuple(CanonicalRational.from_fraction(v) for v in row)
                for row in result.matrix
            )
        )
        if self.dimension != len(expected_matrix.entries):
            raise ValueError("dimension must match the Hankel matrix size")
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
    coefficients: RecurrenceCoefficients
    complete: Literal[True] = True
    method: Literal["EXACT_GRAM_SCHMIDT"] = "EXACT_GRAM_SCHMIDT"

    @model_validator(mode="after")
    def bind_recurrence(self) -> Self:
        from jacobian.math.moments_orthogonal.operations import (
            recurrence_coefficients,
        )

        result = recurrence_coefficients(_to_fractions(self.moments))
        if self.coefficients != result:
            raise ValueError("coefficients must be the exact recurrence coefficients")
        return self


# ---------------------------------------------------------------------------
# Jacobi matrix
# ---------------------------------------------------------------------------


class JacobiMatrixRequest(StrictModel):
    coefficients: RecurrenceCoefficients


class JacobiMatrixResult(JacobiMatrixRequest):
    diagonal: tuple[CanonicalRational, ...]
    off_diagonal: tuple[CanonicalRational, ...]
    complete: Literal[True] = True
    method: Literal["EXACT_TRIDIAGONAL_ASSEMBLY"] = "EXACT_TRIDIAGONAL_ASSEMBLY"

    @model_validator(mode="after")
    def bind_jacobi(self) -> Self:
        from jacobian.math.moments_orthogonal.operations import jacobi_matrix

        result = jacobi_matrix(self.coefficients)
        if self.diagonal != _from_fractions(result.diagonal):
            raise ValueError("diagonal must match the exact Jacobi diagonal")
        if self.off_diagonal != _from_fractions(result.off_diagonal):
            raise ValueError("off_diagonal must match the exact Jacobi off-diagonal")
        return self


# ---------------------------------------------------------------------------
# Christoffel-Darboux kernel
# ---------------------------------------------------------------------------


def _require_combined_kernel_height(
    coefficients: RecurrenceCoefficients,
    x: CanonicalRational,
    y: CanonicalRational,
) -> None:
    """Combined order-and-height admission shared by wire and native callers.

    Each monic recurrence step p_{k+1}(t) = (t - alpha_k) p_k(t)
    - beta_k p_{k-1}(t) raises the evaluated height by at most
    H = height(x/y) + height(alpha) + height(beta) + 2 digits, so p_k(t)
    has height at most k*H and, dividing by h_k = prod(beta), every kernel
    summand has height at most 2*k*H + (k+1)*height(beta). Summing all
    summands bounds the exact kernel; reject any admitted input whose worst
    case could exceed the canonical rational digit limit during execution.
    """
    point_height = max(_coefficient_height(x), _coefficient_height(y))
    alpha_height = max(
        (_coefficient_height(v) for v in coefficients.alpha), default=1
    )
    # The forward recurrence consumes only beta[:len(alpha)]; a trailing
    # mu_n entry never enters the kernel, so it must not drive admission.
    beta_height = max(
        (_coefficient_height(v) for v in coefficients.beta[: len(coefficients.alpha)]),
        default=1,
    )
    step = point_height + alpha_height + beta_height + 2
    kernel_bound = sum(
        2 * k * step + (k + 1) * beta_height + 2 for k in range(len(coefficients.alpha))
    )
    if (
        kernel_bound + len(str(max(len(coefficients.alpha), 1)))
        > MAX_CANONICAL_RATIONAL_DIGITS
    ):
        raise ValueError(
            "Christoffel-Darboux coefficients exceed the conservative "
            "combined order-and-height bound for an exact kernel"
        )


class ChristoffelDarbouxRequest(StrictModel):
    coefficients: RecurrenceCoefficients
    x: CanonicalRational
    y: CanonicalRational

    @model_validator(mode="after")
    def require_valid_coefficients(self) -> Self:
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
        _require_combined_kernel_height(
            self.coefficients, self.x, self.y
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
            self.coefficients,
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
    coefficients: RecurrenceCoefficients

    @model_validator(mode="after")
    def require_valid_coefficients(self) -> Self:
        # Individual finite-float bounds do not bound their combination: a
        # diagonal scale far above the subdiagonals numerically disconnects
        # the Jacobi matrix, localized eigenvector first components
        # underflow, and the rule would lose mass. Golub-Welsch on an
        # admitted request is deterministic and bounded (<= 16 points), and
        # its kernel owns the point-count, underflow, and magnitude bounds,
        # so admission replays it and rejects any recurrence whose weights
        # do not stay strictly positive.
        from jacobian.math.moments_orthogonal.operations import gaussian_quadrature

        gaussian_quadrature(self.coefficients)
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

        result = gaussian_quadrature(self.coefficients)
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
