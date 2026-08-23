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
from jacobian.math.matrices.values import RationalMatrix
from jacobian.math.moments_orthogonal.values import (
    MAX_MOMENTS,
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
    matrix: RationalMatrix
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
        if self.matrix.entries != expected_matrix:
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
        # The kernel inspects only the first 2 * MAX_RECURRENCE_ORDER + 1
        # moments; a longer retained sequence would carry trailing entries
        # whose positivity was never verified.
        if len(self.moments) > 2 * MAX_RECURRENCE_ORDER + 1:
            raise ValueError(
                f"moment sequence length {len(self.moments)} exceeds the "
                f"{2 * MAX_RECURRENCE_ORDER + 1} moments consumed by the "
                "maximum supported recurrence order"
            )
        # The Gram-Schmidt kernel requires a positive-definite moment
        # functional; admit exactly the sequences it accepts so an accepted
        # request cannot fail inside execution. The positivity replay also
        # yields the exact coefficients, so admission can prove the returned
        # result fits the canonical limit before the operation runs.
        from jacobian._exact import CanonicalRational
        from jacobian.math._rational_height import RationalHeight
        from jacobian.math.moments_orthogonal.operations import (
            recurrence_coefficients,
        )

        computed = recurrence_coefficients(_to_fractions(self.moments))
        for value in (*computed.alpha, *computed.beta):
            canonical = CanonicalRational.from_fraction(value)
            if RationalHeight.from_canonical(canonical).exceeds(
                MAX_CANONICAL_RATIONAL_DIGITS
            ):
                raise ValueError(
                    "recurrence coefficient growth exceeds the canonical "
                    f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit result limit; "
                    "reduce the moment component magnitude"
                )
        return self


class RecurrenceCoefficients(StrictModel):
    """The domain-owned three-term recurrence coefficient pair."""

    alpha: tuple[CanonicalRational, ...] = Field(min_length=0)
    beta: tuple[CanonicalRational, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_coefficients(self) -> Self:
        _validate_alpha_beta(self.alpha, self.beta)
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
        expected = RecurrenceCoefficients(
            alpha=_from_fractions(result.alpha),
            beta=_from_fractions(result.beta),
        )
        if self.coefficients != expected:
            raise ValueError("coefficients must be the exact recurrence coefficients")
        return self


# ---------------------------------------------------------------------------
# Jacobi matrix
# ---------------------------------------------------------------------------


class JacobiMatrixRequest(StrictModel):
    coefficients: RecurrenceCoefficients

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
    conservative upper bound from the concrete request (coefficient sizes,
    evaluation-point sizes, and order) and reject any request whose kernel or
    evaluated polynomials could exceed the canonical limit.
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
        # point derives its own bounds so coefficient sizes are charged at
        # every step.
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
        term_bounds.append(
            (px_num[k] + py_num[k] + h_den, px_den[k] + py_den[k] + h_num)
        )
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
    coefficients: RecurrenceCoefficients
    x: CanonicalRational
    y: CanonicalRational

    @model_validator(mode="after")
    def require_valid_coefficients(self) -> Self:
        _validate_alpha_beta(self.coefficients.alpha, self.coefficients.beta)
        require_bounded_rational(self.x, max_digits=MAX_RATIONAL_DIGITS, label="x")
        require_bounded_rational(self.y, max_digits=MAX_RATIONAL_DIGITS, label="y")
        _require_bounded_kernel_growth(
            tuple(v.as_fraction() for v in self.coefficients.alpha),
            tuple(v.as_fraction() for v in self.coefficients.beta),
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
    coefficients: RecurrenceCoefficients

    @model_validator(mode="after")
    def require_valid_coefficients(self) -> Self:
        from jacobian.math.moments_orthogonal.operations import (
            _require_finite_double_coefficients,
        )

        alpha = self.coefficients.alpha
        beta = self.coefficients.beta
        if not 1 <= len(alpha) <= MAX_QUADRATURE_POINTS:
            raise ValueError("alpha must contain between 1 and 16 entries")
        if len(beta) != len(alpha) and len(beta) != len(alpha) + 1:
            raise ValueError("beta must have length len(alpha) or len(alpha)+1")
        for value in (*alpha, *beta):
            require_bounded_rational(
                value, max_digits=MAX_RATIONAL_DIGITS, label="coefficient"
            )
        # The finite-double domain is owned by the Golub-Welsch kernel; the
        # request admits exactly what the native kernel accepts (positivity,
        # underflow bound on beta_0 and each subdiagonal, finite magnitude) so
        # an accepted request cannot fail inside execution.
        _require_finite_double_coefficients(
            tuple(value.as_fraction() for value in alpha),
            tuple(value.as_fraction() for value in beta),
        )
        return self


class GaussianQuadratureResult(GaussianQuadratureRequest):
    """Approximate Gaussian quadrature result via Golub-Welsch (IEEE double).

    Nodes are eigenvalues of the Jacobi matrix (generally irrational, e.g.
    ``alpha=(0,0), beta=(1,2)`` has exact nodes ``±sqrt(2)``). The
    decomposition runs in IEEE doubles, so ``approximate_nodes`` and
    ``approximate_weights`` are **approximations** with double precision. Each
    entry is the exact dyadic rational image of the computed double for
    canonical JSON transport; the result is explicitly approximate.
    """

    approximate_nodes: tuple[CanonicalRational, ...] = Field(
        description=(
            "Approximate quadrature nodes as dyadic rationals imaging IEEE doubles"
            " (not exact algebraic numbers; e.g. ±sqrt(2) is approximated)."
        )
    )
    approximate_weights: tuple[CanonicalRational, ...] = Field(
        description=(
            "Approximate quadrature weights as dyadic rationals imaging IEEE doubles."
        )
    )
    method: Literal["GOLUB_WELSCH_APPROXIMATE"] = "GOLUB_WELSCH_APPROXIMATE"
    approximation: Literal["IEEE_DOUBLE"] = "IEEE_DOUBLE"

    @model_validator(mode="after")
    def bind_gaussian_quadrature(self) -> Self:
        from jacobian.math.moments_orthogonal.operations import (
            gaussian_quadrature,
        )

        result = gaussian_quadrature(
            _to_fractions(self.coefficients.alpha),
            _to_fractions(self.coefficients.beta),
        )
        if self.approximate_nodes != _from_fractions(result.approximate_nodes):
            raise ValueError(
                "approximate_nodes must match the Golub-Welsch eigenvalues"
            )
        if self.approximate_weights != _from_fractions(result.approximate_weights):
            raise ValueError("approximate_weights must match the Golub-Welsch weights")
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
    "RecurrenceCoefficients",
    "RecurrenceCoefficientsRequest",
    "RecurrenceCoefficientsResult",
]
