"""Typed wire contracts for moment-functional operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math._rational_height import RationalHeight
from jacobian.math.moments_orthogonal.values import (
    MAX_HANKEL_ORDER,
    MAX_POLYNOMIAL_DEGREE,
    MomentFunctionalPrefix,
    OrthogonalPolynomialFamily,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"moments_orthogonal.{reason}", message)


def _require_moment_height(
    prefix: MomentFunctionalPrefix, count: int, bound: int, reason: str, message: str
) -> None:
    """Apply a request-local exact-height envelope without running a kernel."""
    if any(
        RationalHeight.from_canonical(value).exceeds(bound)
        for value in prefix.moments[:count]
    ):
        raise _validation_error(reason, message)


class HankelRequest(StrictModel):
    """Compute the Hankel matrix H_r from a moment prefix."""

    prefix: MomentFunctionalPrefix
    order: int = Field(ge=0, le=MAX_HANKEL_ORDER)

    @model_validator(mode="after")
    def require_sufficient_moments(self) -> Self:
        needed = 2 * self.order + 1
        if len(self.prefix.moments) < needed:
            raise _validation_error(
                "insufficient_moments",
                f"need at least {needed} moments for order {self.order}, got {len(self.prefix.moments)}",
            )
        bound = max(32_768 // ((self.order + 1) ** 2) - 2, 8)
        _require_moment_height(
            self.prefix,
            needed,
            bound,
            "determinant_height",
            f"moment heights exceed the conservative {bound}-digit bound for an exact order-{self.order} determinant",
        )
        return self


class ShiftedHankelRequest(StrictModel):
    """Compute the shifted Hankel matrix H_r^(1)[i,j] = mu_(i+j+1)."""

    prefix: MomentFunctionalPrefix
    # A shifted matrix of order r consumes mu_1..mu_(2r+1); the canonical
    # prefix holds at most 65 moments, so r = 32 could never validate and
    # must not be advertised as supported.
    order: int = Field(ge=0, le=MAX_HANKEL_ORDER - 1)

    @model_validator(mode="after")
    def require_sufficient_moments(self) -> Self:
        needed = 2 * self.order + 2
        if len(self.prefix.moments) < needed:
            raise _validation_error(
                "insufficient_moments",
                f"need at least {needed} moments for shifted order {self.order}, got {len(self.prefix.moments)}",
            )
        bound = max(32_768 // ((self.order + 1) ** 2) - 2, 8)
        _require_moment_height(
            MomentFunctionalPrefix(
                moments=self.prefix.moments[1:], variable=self.prefix.variable
            ),
            2 * self.order + 1,
            bound,
            "determinant_height",
            f"moment heights exceed the conservative {bound}-digit bound for an exact order-{self.order} determinant",
        )
        return self


class OrthogonalPolynomialRequest(StrictModel):
    """Compute monic orthogonal polynomials from moments."""

    prefix: MomentFunctionalPrefix
    max_degree: int = Field(ge=0, le=MAX_POLYNOMIAL_DEGREE)

    @model_validator(mode="after")
    def require_sufficient_moments(self) -> Self:
        needed = 2 * self.max_degree + 1
        if len(self.prefix.moments) < needed:
            raise _validation_error(
                "insufficient_moments",
                f"need at least {needed} moments for degree {self.max_degree}, got {len(self.prefix.moments)}",
            )
        return self

    @model_validator(mode="after")
    def require_gram_schmidt_height(self) -> Self:
        """Bound only the request envelope; execution remains owner-local."""
        side = self.max_degree + 1
        bound = max((32_768 - 2 * side) // (2 * side * (side + 1)), 8)
        _require_moment_height(
            self.prefix,
            2 * self.max_degree + 1,
            bound,
            "gram_schmidt_height",
            f"moment heights exceed the conservative {bound}-digit bound for exact degree-{self.max_degree} Gram-Schmidt; supply a smaller or better-scaled moment prefix",
        )
        return self


class RecurrenceRequest(StrictModel):
    """Compute three-term recurrence coefficients from a family."""

    family: OrthogonalPolynomialFamily

    @model_validator(mode="after")
    def require_quasi_definite_family(self) -> Self:
        """The kernel divides by every squared norm except the terminal one
        (``beta_k = h_k / h_{k-1}`` for k >= 1); a vanishing interior norm
        would leak ZeroDivisionError instead of a typed result, while a
        vanishing terminal norm leaves ``alpha`` and every ``beta`` exactly
        defined. Admission then replays the exact derivation so every
        emitted ratio is height-checked here — a family such as
        h_0 = 10^-20000 with h_1 = 10^20000 (beta_1 = 10^40000) fails
        parsing, not execution.
        """
        interior = self.family.polynomials[:-1]
        if any(term.squared_norm.as_fraction() == 0 for term in interior):
            raise _validation_error(
                "nonzero_norm",
                "recurrence coefficients require every non-terminal "
                "squared norm to be nonzero",
            )
        return self


class ChristoffelDarbouxRequest(StrictModel):
    """Compute the Christoffel-Darboux kernel."""

    family: OrthogonalPolynomialFamily
    degree: int = Field(ge=0)

    @model_validator(mode="after")
    def require_degree_within_family(self) -> Self:
        if self.degree >= len(self.family.polynomials):
            raise _validation_error(
                "degree_out_of_range",
                f"kernel degree {self.degree} exceeds the supplied family "
                f"of {len(self.family.polynomials)} polynomials",
            )
        return self

    @model_validator(mode="after")
    def require_nonzero_norms_through_degree(self) -> Self:
        """The defining sum divides only by norms through this degree."""
        for term in self.family.polynomials[: self.degree + 1]:
            if term.squared_norm.as_fraction() == 0:
                raise _validation_error(
                    "nonzero_norm",
                    f"Christoffel-Darboux kernel degree {self.degree} "
                    f"requires nonzero squared norms through degree "
                    f"{self.degree}, but p_{term.degree} has a vanishing norm",
                )
        return self


class JacobiMatrixRequest(StrictModel):
    """Compute the finite Jacobi matrix."""

    family: OrthogonalPolynomialFamily

    @model_validator(mode="after")
    def require_representable_recurrence(self) -> Self:
        """Pre-computation height bound on the derived recurrence entries.

        Every derived alpha is a difference of admitted family
        coefficients, but representable coefficients do not imply a
        representable difference; each derived beta is an adjacent-norm
        ratio h_k / h_{k-1}: a vanishing norm makes the ratio undefined
        and a representable pair can still exceed the canonical result
        height. Reject such families here so every accepted request can
        return its declared result.
        """
        from jacobian.math.moments_orthogonal._jacobi import (
            JacobiMatrixAdmissionError,
            require_jacobi_matrix_admission,
        )

        try:
            require_jacobi_matrix_admission(self.family)
        except JacobiMatrixAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
        return self


class GaussianQuadratureRequest(StrictModel):
    """Compute an exact Gaussian quadrature rule."""

    prefix: MomentFunctionalPrefix
    order: int = Field(ge=1, le=16)

    @model_validator(mode="after")
    def require_sufficient_moments(self) -> Self:
        needed = 2 * self.order
        if len(self.prefix.moments) < needed:
            raise _validation_error(
                "insufficient_moments",
                f"need at least {needed} moments for quadrature order {self.order}, got {len(self.prefix.moments)}",
            )
        side = self.order + 1
        bound = max((32_768 - 2 * side) // (2 * side * (side + 1)), 8)
        _require_moment_height(
            self.prefix,
            needed,
            bound,
            "gram_schmidt_height",
            f"moment heights exceed the conservative {bound}-digit bound for exact degree-{self.order} Gram-Schmidt; supply a smaller or better-scaled moment prefix",
        )
        return self


__all__ = [
    "ChristoffelDarbouxRequest",
    "GaussianQuadratureRequest",
    "HankelRequest",
    "JacobiMatrixRequest",
    "OrthogonalPolynomialRequest",
    "RecurrenceRequest",
    "ShiftedHankelRequest",
]


MomentFunctionalPrefix.model_rebuild()
HankelRequest.model_rebuild()
ShiftedHankelRequest.model_rebuild()
OrthogonalPolynomialRequest.model_rebuild()
ChristoffelDarbouxRequest.model_rebuild()
GaussianQuadratureRequest.model_rebuild()
