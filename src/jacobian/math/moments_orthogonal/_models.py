"""Typed wire contracts for exact moments and orthogonal polynomials."""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, ValidationError, model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
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

# Per-entry conversion bounds do not bound the conditioning of the assembled
# Jacobi matrix: entries spanning many magnitudes produce eigenvalues and
# eigenvector components that vanish in float64 although they are
# mathematically nonzero. The assembled nonzero entries must share one
# magnitude scale, and the exact determinant must exclude eigenvalues whose
# only representation is below the float64 underflow floor. With n <= 16 and
# entry spread <= R, an eigenvector first component is bounded below by
# |q_1| >= 1 / (n * (5R)^(n-1)), so these constants keep every Golub-Welsch
# weight above QUADRATURE_WEIGHT_UNDERFLOW_FLOOR and every nonzero node above
# the double-precision underflow floor.
MAX_QUADRATURE_ENTRY_SPREAD = Fraction(10) ** 6
QUADRATURE_WEIGHT_UNDERFLOW_FLOOR = Fraction(1, 10**280)
QUADRATURE_NODE_UNDERFLOW_FLOOR = Fraction(1, 10**280)


def _to_fractions(
    values: tuple[CanonicalRational, ...],
) -> tuple[Fraction, ...]:
    return tuple(v.as_fraction() for v in values)


def _from_fractions(
    values: Iterable[Fraction],
) -> tuple[CanonicalRational, ...]:
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
        # The kernel consumes at most 2*MAX_RECURRENCE_ORDER moments: order
        # min(MAX_RECURRENCE_ORDER, (m-1)//2) coefficients need an odd moment
        # count, and 33 moments would derive a seventeenth beta entry outside
        # the canonical RecurrenceCoefficientsValue domain.
        if len(self.moments) > 2 * MAX_RECURRENCE_ORDER:
            raise ValueError(
                f"moment sequence length {len(self.moments)} exceeds the "
                f"{2 * MAX_RECURRENCE_ORDER} moments consumed by the maximum "
                "supported recurrence order"
            )
        # A nonpositive zeroth moment is not a positive functional, and the
        # kernel's short-sequence return would otherwise emit beta_0 = mu_0
        # before any positive-definiteness check.
        if self.moments[0].as_fraction() <= 0:
            raise ValueError(
                "the zeroth moment must be positive for a positive functional"
            )
        # The Gram-Schmidt kernel requires a positive-definite moment
        # functional; admit exactly the sequences it accepts so an accepted
        # request cannot fail inside execution.
        from jacobian.math.moments_orthogonal.operations import (
            recurrence_coefficients,
        )

        derived = recurrence_coefficients(_to_fractions(self.moments))
        # Derived-coefficient growth budget: the typed result carries alpha
        # and beta as canonical rationals inside RecurrenceCoefficientsValue,
        # so admission must reject sequences whose exact Gram-Schmidt output
        # leaves that canonical domain.
        try:
            RecurrenceCoefficientsValue(
                alpha=_from_fractions(derived.alpha),
                beta=_from_fractions(derived.beta),
            )
        except ValidationError as exc:
            raise ValueError(
                "derived recurrence coefficients exceed the canonical "
                "coefficient value domain"
            ) from exc
        return self


class RecurrenceCoefficientsValue(StrictModel):
    """The one canonical recurrence-coefficient value.

    Produced by ``moments.recurrence_coefficients.compute`` and accepted
    unchanged by the Jacobi-matrix, Christoffel-Darboux, and quadrature
    consumers, so a serialized result composes into those requests without
    reconstructing parallel payloads.
    """

    alpha: tuple[CanonicalRational, ...]
    beta: tuple[CanonicalRational, ...]

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_alpha_beta(self.alpha, self.beta)
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
                "coefficients must be the exact recurrence coefficients of "
                "the retained moments"
            )
        return self


# ---------------------------------------------------------------------------
# Jacobi matrix
# ---------------------------------------------------------------------------


class JacobiMatrixRequest(StrictModel):
    coefficients: RecurrenceCoefficientsValue


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
        alpha = self.coefficients.alpha
        beta = self.coefficients.beta
        # Coefficient magnitude grows with recurrence order (p_{k+1}(z) =
        # (z - alpha_k) p_k(z) - beta_k p_{k-1}(z)), so bounding x and y alone
        # cannot bound polynomials_evaluated or the kernel. Run the exact
        # bounded recurrence here and admit exactly the requests whose every
        # returned component fits the canonical rational limit, mirroring the
        # admit-what-the-kernel-accepts contract of the moment-sequence
        # requests in this module.
        from jacobian.math.moments_orthogonal.operations import (
            christoffel_darboux,
        )

        result = christoffel_darboux(
            _to_fractions(alpha),
            _to_fractions(beta),
            self.x.as_fraction(),
            self.y.as_fraction(),
        )
        for value in (result.kernel, *result.polynomials_evaluated):
            if (
                len(format_canonical_integer(value.numerator))
                > MAX_CANONICAL_RATIONAL_DIGITS
                or len(format_canonical_integer(value.denominator))
                > MAX_CANONICAL_RATIONAL_DIGITS
            ):
                raise ValueError(
                    "requested kernel order exceeds the canonical "
                    f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit component limit"
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


def _require_admissible_quadrature_entry(value: CanonicalRational) -> None:
    """Bound one recurrence entry to values that survive float64 conversion."""
    require_bounded_rational(value, max_digits=MAX_RATIONAL_DIGITS, label="coefficient")
    magnitude = abs(value.as_fraction())
    if magnitude > MAX_QUADRATURE_MAGNITUDE:
        raise ValueError(
            "quadrature coefficients exceed the finite-float magnitude bound"
        )
    # A semantically nonzero coefficient that converts to 0.0 would
    # silently collapse a node or weight to zero; every admitted
    # nonzero entry must survive double conversion as itself.
    if 0 < magnitude < MIN_QUADRATURE_SUBDIAGONAL:
        raise ValueError(
            "quadrature coefficients below the underflow bound would "
            "convert to zero doubles"
        )


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
                "beta_0 (the zeroth moment of a positive functional) must be positive"
            )
        if beta_zero < MIN_QUADRATURE_SUBDIAGONAL:
            raise ValueError(
                "beta_0 falls below the quadrature underflow bound and would give zero weight"
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
            if sub < MIN_QUADRATURE_SUBDIAGONAL:
                raise ValueError(
                    "subdiagonal beta entries fall below the quadrature underflow bound"
                )
        for value in (*alpha, *beta):
            _require_admissible_quadrature_entry(value)
        self._require_conditioned_jacobi_matrix(alpha, beta)
        return self

    @staticmethod
    def _require_conditioned_jacobi_matrix(
        alpha: tuple[CanonicalRational, ...],
        beta: tuple[CanonicalRational, ...],
    ) -> None:
        """Reject Jacobi matrices whose float64 spectrum cannot be resolved.

        The per-entry bounds above admit matrices whose eigen-decomposition
        underflows: alpha=(0, 10^300) with beta=(1, 10^-300) assembles
        [[0, 10^-150], [10^-150, 10^300]], whose small exact eigenvalue and
        its eigenvector components vanish in double precision although the
        matrix is irreducible and both Gaussian weights are positive.
        """
        alpha_magnitudes = [
            abs(value.as_fraction()) for value in alpha if value.as_fraction() != 0
        ]
        beta_magnitudes = [
            value.as_fraction() for value in beta if value.as_fraction() != 0
        ]
        magnitudes = alpha_magnitudes + beta_magnitudes
        if magnitudes:
            spread = max(magnitudes) / min(magnitudes)
            if spread > MAX_QUADRATURE_ENTRY_SPREAD:
                raise ValueError(
                    "assembled quadrature entries span more than "
                    f"{MAX_QUADRATURE_ENTRY_SPREAD} magnitudes; the "
                    "Golub-Welsch spectrum would underflow float64"
                )
            # Weight floor: with entry spread R and n <= 16, every unit
            # eigenvector first component satisfies |q_1| >= 1/(n*(5R)^(n-1)),
            # so the smallest representable weight is bounded below by
            # beta_0 / (n^2 * (5R)^(2n-2)).
            count = len(alpha)
            weight_floor = beta_magnitudes[0] / (
                Fraction(count * count) * (5 * spread) ** (2 * (count - 1))
            )
            if weight_floor < QUADRATURE_WEIGHT_UNDERFLOW_FLOOR:
                raise ValueError(
                    "quadrature weights may underflow float64 for this "
                    "Jacobi matrix conditioning"
                )
        # Node floor: leading principal minors of a tridiagonal matrix with
        # squared off-diagonal beta satisfy the rational recurrence
        # D_k = alpha_k * D_{k-1} - beta_k * D_{k-2}, so det is exact. A
        # nonzero determinant smaller than the underflow floor times the
        # norm bound means some eigenvalue exists only below float64.
        previous_previous = Fraction(1)
        previous = alpha[0].as_fraction()
        for index in range(1, len(alpha)):
            current = (
                alpha[index].as_fraction() * previous
                - beta[index].as_fraction() * previous_previous
            )
            previous_previous, previous = previous, current
        determinant = previous
        scale = (
            max(
                max((abs(value.as_fraction()) for value in alpha), default=Fraction(0)),
                max(beta_magnitudes, default=Fraction(0)),
            )
            + 1
        )
        node_floor = QUADRATURE_NODE_UNDERFLOW_FLOOR * (2 * scale) ** (len(alpha) - 1)
        if determinant != 0 and abs(determinant) < node_floor:
            raise ValueError(
                "the assembled Jacobi matrix has an exact eigenvalue below "
                "the float64 underflow floor; Golub-Welsch would return a "
                "vanishing node for a mathematically nonzero one"
            )


class GaussianQuadratureResult(GaussianQuadratureRequest):
    nodes: tuple[CanonicalRational, ...]
    weights: tuple[CanonicalRational, ...]
    complete: Literal[True] = True
    method: Literal["APPROXIMATE_GOLUB_WELSCH_FLOAT64"] = (
        "APPROXIMATE_GOLUB_WELSCH_FLOAT64"
    )
    approximation: Literal["FLOAT64_ROUNDED_DYADIC_RATIONAL"] = (
        "FLOAT64_ROUNDED_DYADIC_RATIONAL"
    )

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
    "RecurrenceCoefficientsValue",
]
