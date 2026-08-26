"""Typed wire contracts for moment-functional operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.moments_orthogonal.values import (
    MAX_HANKEL_ORDER,
    MAX_POLYNOMIAL_DEGREE,
    MomentFunctionalPrefix,
    OrthogonalPolynomialFamily,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"moments_orthogonal.{reason}", message)


class HankelRequest(StrictModel):
    """Compute the Hankel matrix H_r from a moment prefix."""

    prefix: MomentFunctionalPrefix
    order: int = Field(ge=0, le=MAX_HANKEL_ORDER)

    @model_validator(mode="after")
    def require_sufficient_moments(self) -> Self:
        from jacobian.math.moments_orthogonal.operations import (
            HankelMatrixAdmissionError,
            require_hankel_matrix_admission,
        )

        try:
            require_hankel_matrix_admission(self.prefix, self.order, shifted=False)
        except HankelMatrixAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
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
        from jacobian.math.moments_orthogonal.operations import (
            HankelMatrixAdmissionError,
            require_hankel_matrix_admission,
        )

        try:
            require_hankel_matrix_admission(self.prefix, self.order, shifted=True)
        except HankelMatrixAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
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
    def require_quasi_definite_prefix(self) -> Self:
        """Replay the exact Gram-Schmidt kernel so a prefix whose orthogonal
        family would hit a zero squared norm is rejected at the boundary
        instead of failing inside execution.

        The conservative height gate runs FIRST: without it, parsing would
        perform every exact projection on unbounded intermediates before
        discovering an over-tall family at wire construction. After the
        gate, both this admission replay and the execution that follows it
        operate on provably bounded intermediates with typed height checks.
        """
        from jacobian.math.moments_orthogonal.operations import (
            MomentsOrthogonalAdmissionError,
            _require_gram_schmidt_heights_admissible,
        )

        try:
            _require_gram_schmidt_heights_admissible(
                self.prefix.moments, self.max_degree
            )
        except MomentsOrthogonalAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
        from jacobian.math.moments_orthogonal.operations import (
            orthogonal_polynomials_from_moments,
        )

        orthogonal_polynomials_from_moments(
            [_m.as_fraction() for _m in self.prefix.moments],
            self.max_degree,
            self.prefix.variable,
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
        from jacobian.math.moments_orthogonal.operations import compute_recurrence

        compute_recurrence(self)
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
        """The defining sum divides each p_k(x) p_k(y) term by h_k; only
        norms through the requested degree are consumed and gate admission.
        Admission then replays the bounded coefficient construction so an
        over-tall kernel (e.g. p_1 = x + 10^17000 with unit norms at
        degree 1, whose constant coefficient reaches 10^34000 + 1) fails
        parsing instead of raising during execution.
        """
        for term in self.family.polynomials[: self.degree + 1]:
            if term.squared_norm.as_fraction() == 0:
                raise _validation_error(
                    "nonzero_norm",
                    f"Christoffel-Darboux kernel degree {self.degree} "
                    f"requires nonzero squared norms through degree "
                    f"{self.degree}, but p_{term.degree} has a vanishing norm",
                )
        from jacobian.math.moments_orthogonal.operations import (
            compute_christoffel_darboux,
        )

        compute_christoffel_darboux(self)
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
        from jacobian.math.moments_orthogonal.operations import (
            GaussianQuadratureAdmissionError,
            require_gaussian_quadrature_admission,
        )

        try:
            require_gaussian_quadrature_admission(self.prefix, self.order)
        except GaussianQuadratureAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
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
