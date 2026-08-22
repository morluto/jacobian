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
# coefficient must convert to a finite double, every semantically nonzero
# coefficient must stay clear of underflow so its mathematical value survives
# conversion, and subdiagonal entries must stay far from both boundaries so
# their square roots are exact enough.
MAX_QUADRATURE_MAGNITUDE = Fraction(10) ** 300
MIN_QUADRATURE_MAGNITUDE = Fraction(1, 10 ** 300)


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


def _digits(count: int) -> int:
    return len(str(abs(count)))


def _require_bounded_kernel_growth(
    alpha: tuple[Fraction, ...],
    beta: tuple[Fraction, ...],
    x: Fraction,
    y: Fraction,
) -> None:
    """Bound recurrence and kernel digit growth before the backend runs.

    The forward recurrence p_{k+1}(t) = (t - alpha_k) p_k(t) - beta_k p_{k-1}(t)
    compounds numerator and denominator digits across steps; derive a
    conservative upper bound from the concrete request and reject any request
    whose kernel or evaluated polynomials could exceed the canonical limit.
    """
    limit = MAX_CANONICAL_RATIONAL_DIGITS

    def parts(value: Fraction) -> tuple[int, int]:
        return _digits(value.numerator), _digits(value.denominator)

    xn, xd = parts(x)
    yn, yd = parts(y)
    an = [_digits(v.numerator) for v in alpha]
    ad = [_digits(v.denominator) for v in alpha]
    bn = [_digits(v.numerator) for v in beta]
    bd = [_digits(v.denominator) for v in beta]

    def recurrence(point_n: int, point_d: int) -> tuple[list[int], list[int]]:
        # Digit bounds (numerator, denominator) of p_k evaluated at one point;
        # no reduction credit is taken, so these are strict upper bounds.
        pn = [0]
        pd = [0]
        if not alpha:
            return pn, pd
        # u_k = point - alpha_k over a common denominator; each evaluation
        # point derives its own bounds so a large y is fully charged.
        un = [max(point_n + ad[k], an[k] + point_d) + 1 for k in range(len(alpha))]
        ud = [point_d + ad[k] for k in range(len(alpha))]
        pn.append(un[0])
        pd.append(ud[0])
        for k in range(1, len(alpha)):
            w_num, w_den = bn[k], bd[k]
            pn.append(
                max(
                    un[k] + pn[k] + w_den + pd[k - 1],
                    w_num + pn[k - 1] + ud[k] + pd[k],
                )
                + 1
            )
            pd.append(ud[k] + pd[k] + w_den + pd[k - 1])
        return pn, pd

    px_num, px_den = recurrence(xn, xd)
    py_num, py_den = recurrence(yn, yd)

    # The kernel evaluates p_0..p_{n-1} and advances h by beta[1..n-1].
    for k in range(len(alpha)):
        if max(px_num[k], px_den[k], py_num[k], py_den[k]) > limit:
            raise ValueError(
                "Christoffel-Darboux recurrence growth exceeds the canonical "
                f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit result limit; reduce "
                "coefficient or evaluation-point magnitude"
            )

    h_num = 0
    h_den = 0
    total_den = 0
    term_bounds: list[tuple[int, int]] = []
    for k in range(len(alpha)):
        h_num += bn[k]
        h_den += bd[k]
        # term_k = p_k(x) p_k(y) / h_k before reduction.
        term_bounds.append((px_num[k] + py_num[k] + h_den,
                            px_den[k] + py_den[k] + h_num))
        total_den += term_bounds[-1][1]
    # Summing over the common denominator multiplies each term numerator by
    # every other term's denominator; reduction only shrinks the result.
    kernel_num = (
        max(term_num + total_den - term_den for term_num, term_den in term_bounds)
        + len(term_bounds).bit_length()
        + 1
        if term_bounds
        else 1
    )
    if max(kernel_num, total_den) > limit:
        raise ValueError(
            "Christoffel-Darboux kernel growth exceeds the canonical "
            f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit result limit; reduce "
            "coefficient or evaluation-point magnitude"
        )


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
        _require_bounded_kernel_growth(
            tuple(v.as_fraction() for v in self.alpha),
            tuple(v.as_fraction() for v in self.beta),
            self.x.as_fraction(),
            self.y.as_fraction(),
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
        # Subdiagonal entries feed math.sqrt after float conversion; they must
        # be positive squared-norm ratios.
        for index in range(1, min(len(self.alpha), len(self.beta))):
            if self.beta[index].as_fraction() <= 0:
                raise ValueError(
                    "subdiagonal beta entries must be positive squared-norm ratios"
                )
        # Every coefficient becomes an IEEE double; a semantically nonzero
        # value that underflows to 0.0 would erase positive quadrature mass or
        # collapse node positions, so it must survive conversion.
        for value in (*self.alpha, *self.beta):
            require_bounded_rational(
                value, max_digits=MAX_RATIONAL_DIGITS, label="coefficient"
            )
            converted = value.as_fraction()
            magnitude = abs(converted)
            if magnitude > MAX_QUADRATURE_MAGNITUDE:
                raise ValueError(
                    "quadrature coefficients exceed the finite-float magnitude bound"
                )
            if converted != 0 and magnitude < MIN_QUADRATURE_MAGNITUDE:
                raise ValueError(
                    "quadrature coefficients fall below the finite-float "
                    "underflow bound"
                )
        return self


class GaussianQuadratureResult(GaussianQuadratureRequest):
    nodes: tuple[CanonicalRational, ...]
    weights: tuple[CanonicalRational, ...]
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
