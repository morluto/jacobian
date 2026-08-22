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

    @model_validator(mode="after")
    def require_representable_recurrence(self) -> Self:
        """Pre-computation height bound on the derived recurrence entries.

        Every derived alpha is a difference of admitted family
        coefficients, but each derived beta is an adjacent-norm ratio
        h_k / h_{k-1}, whose height can exceed the canonical result type
        even when both norms fit. Reject such families here so every
        accepted request can return its declared result.
        """
        polys = self.family.polynomials
        # A canonical rational carries at most MAX_CANONICAL_RATIONAL_DIGITS
        # digits, i.e. its absolute value stays below 10**that limit.
        digit_limit = 10**MAX_CANONICAL_RATIONAL_DIGITS
        for k in range(1, len(polys)):
            h_k = polys[k].squared_norm.as_fraction()
            h_prev = polys[k - 1].squared_norm.as_fraction()
            ratio = h_k / h_prev
            if abs(ratio.numerator) >= digit_limit or ratio.denominator >= digit_limit:
                raise ValueError(
                    f"adjacent-norm ratio beta_{k} exceeds the canonical "
                    "rational digit limit; supply a family whose squared "
                    "norm ratios stay representable"
                )
        return self


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
            sympy.Rational(*c.as_integer_ratio()) * x**i
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
            [Fraction(*c.as_integer_ratio()) for c in coefficients],
            [Fraction(*v.as_integer_ratio()) for v in self.prefix.moments],
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
