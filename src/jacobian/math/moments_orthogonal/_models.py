"""Typed wire contracts for moment-functional operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
)
from jacobian._models import StrictModel
from jacobian.math._rational_height import RationalHeight
from jacobian.math.moments_orthogonal.values import (
    MAX_HANKEL_ORDER,
    MAX_POLYNOMIAL_DEGREE,
    MomentFunctionalPrefix,
    OrthogonalPolynomialFamily,
)


def _require_determinant_representable(moments, order: int) -> None:
    """Bound entry heights so the exact determinant stays canonical.

    The determinant of an (order+1)-square rational matrix carries at most
    roughly (order+1)^2 * H digits for H-digit entries; capping each
    moment's height keeps it inside MAX_CANONICAL_RATIONAL_DIGITS.
    """
    per_entry = MAX_CANONICAL_RATIONAL_DIGITS // ((order + 1) ** 2)
    for value in moments:
        if RationalHeight.from_canonical(value).exceeds(max(per_entry - 2, 8)):
            raise ValueError(
                f"moment heights exceed the conservative {max(per_entry - 2, 8)}-digit "
                f"bound for an exact order-{order} determinant"
            )


class HankelRequest(StrictModel):
    """Compute the Hankel matrix H_r from a moment prefix."""

    prefix: MomentFunctionalPrefix
    order: int = Field(ge=0, le=MAX_HANKEL_ORDER)

    @model_validator(mode="after")
    def require_sufficient_moments(self) -> Self:
        needed = 2 * self.order + 1
        if len(self.prefix.moments) < needed:
            raise ValueError(
                f"need at least {needed} moments for order {self.order}, got {len(self.prefix.moments)}"
            )
        _require_determinant_representable(self.prefix.moments, self.order)
        return self


class ShiftedHankelRequest(StrictModel):
    """Compute the shifted Hankel matrix H_r^(1)[i,j] = mu_(i+j+1)."""

    prefix: MomentFunctionalPrefix
    order: int = Field(ge=0, le=MAX_HANKEL_ORDER)

    @model_validator(mode="after")
    def require_sufficient_moments(self) -> Self:
        needed = 2 * self.order + 2
        if len(self.prefix.moments) < needed:
            raise ValueError(
                f"need at least {needed} moments for shifted order {self.order}, got {len(self.prefix.moments)}"
            )
        _require_determinant_representable(self.prefix.moments, self.order)
        return self


class OrthogonalPolynomialRequest(StrictModel):
    """Compute monic orthogonal polynomials from moments."""

    prefix: MomentFunctionalPrefix
    max_degree: int = Field(ge=0, le=MAX_POLYNOMIAL_DEGREE)

    @model_validator(mode="after")
    def require_sufficient_moments(self) -> Self:
        needed = 2 * self.max_degree + 1
        if len(self.prefix.moments) < needed:
            raise ValueError(
                f"need at least {needed} moments for degree {self.max_degree}, got {len(self.prefix.moments)}"
            )
        return self

    @model_validator(mode="after")
    def require_quasi_definite_prefix(self) -> Self:
        """Replay the exact Gram-Schmidt kernel so a prefix whose orthogonal
        family would hit a zero squared norm is rejected at the boundary
        instead of failing inside execution."""
        from jacobian.math.moments_orthogonal.operations import (
            compute_orthogonal_polynomials,
        )

        compute_orthogonal_polynomials(self)
        return self


class RecurrenceRequest(StrictModel):
    """Compute three-term recurrence coefficients from a family."""

    family: OrthogonalPolynomialFamily


class ChristoffelDarbouxRequest(StrictModel):
    """Compute the Christoffel-Darboux kernel."""

    family: OrthogonalPolynomialFamily
    degree: int = Field(ge=0)

    @model_validator(mode="after")
    def require_degree_within_family(self) -> Self:
        if self.degree >= len(self.family.polynomials):
            raise ValueError(
                f"kernel degree {self.degree} exceeds the supplied family "
                f"of {len(self.family.polynomials)} polynomials"
            )
        return self


class JacobiMatrixRequest(StrictModel):
    """Compute the finite Jacobi matrix."""

    family: OrthogonalPolynomialFamily


class GaussianQuadratureRequest(StrictModel):
    """Compute an exact Gaussian quadrature rule."""

    prefix: MomentFunctionalPrefix
    order: int = Field(ge=1, le=16)

    @model_validator(mode="after")
    def require_sufficient_moments(self) -> Self:
        # Constructing p_order via Gram-Schmidt also computes its squared
        # norm h_order = <p_n, p_n>, which consumes mu_{2n}; the public
        # boundary therefore requires 2n+1 moments exactly like the nested
        # orthogonal-polynomial request.
        needed = 2 * self.order + 1
        if len(self.prefix.moments) < needed:
            raise ValueError(
                f"need at least {needed} moments for quadrature order {self.order}, got {len(self.prefix.moments)}"
            )
        return self

    @model_validator(mode="after")
    def require_rational_nodes(self) -> Self:
        """Admit only prefixes whose degree-n orthogonal polynomial splits
        into distinct linear factors over QQ.

        The exact-node contract carries canonical rationals; algebraic
        nodes such as +-sqrt(1/3) cannot be represented, so such prefixes
        are rejected here instead of failing during execution.
        """
        import sympy

        from jacobian.math.moments_orthogonal.operations import (
            compute_orthogonal_polynomials,
        )

        family = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(prefix=self.prefix, max_degree=self.order)
        )
        coefficients = family.polynomials[self.order].coefficients
        x = sympy.Symbol(self.prefix.variable)
        poly = sum(
            sympy.Rational(int(c.num), int(c.den)) * x**i
            for i, c in enumerate(coefficients)
        )
        _, factors = sympy.factor_list(poly)
        if any(
            sympy.degree(factor, x) != 1 or multiplicity != 1
            for factor, multiplicity in factors
        ):
            raise ValueError(
                f"quadrature order {self.order} requires p_{self.order} to "
                "split into distinct linear factors over QQ so every node "
                "is an exact rational; this moment prefix yields algebraic "
                "or repeated nodes"
            )
        # Positive weights are part of the declared contract; replay the
        # exact construction so a nonpositive weight is rejected here
        # instead of raising during execution.
        from fractions import Fraction

        from jacobian.math.moments_orthogonal.operations import (
            _construct_quadrature_rule,
        )

        _nodes, weights = _construct_quadrature_rule(
            [Fraction(int(c.num), int(c.den)) for c in coefficients],
            [Fraction(int(v.num), int(v.den)) for v in self.prefix.moments],
            self.order,
        )
        if any(weight <= 0 for weight in weights):
            raise ValueError(
                "quadrature admission requires strictly positive weights; "
                "this moment prefix yields a nonpositive weight"
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
