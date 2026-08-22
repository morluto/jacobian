"""Typed wire contracts for polynomial support geometry operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomial_support_geometry.values import (
    MAX_NEWTON_TERMS,
    MAX_WEIGHT_COMPONENTS,
)
from jacobian.math.polynomials.values import RationalPolynomial


class SupportRequest(StrictModel):
    """Request the support of a polynomial."""

    polynomial: RationalPolynomial = Field(
        description=(
            "The canonical nonzero-or-zero sparse polynomial whose exponent "
            "support is requested."
        )
    )


class NewtonPolytopeRequest(StrictModel):
    """Request the Newton polytope of a polynomial."""

    polynomial: RationalPolynomial = Field(
        description=(
            "The canonical sparse polynomial whose Newton polytope is "
            f"requested; at most {MAX_NEWTON_TERMS} terms so the per-point "
            "exact extremality work stays bounded."
        )
    )

    @model_validator(mode="after")
    def require_feasible_hull_work(self) -> Self:
        # Exact extremality testing solves one bounded rational LP per
        # support point against all others; keep the admitted quadratic-ish
        # work conservatively small instead of admitting the full canonical
        # term budget.
        if len(self.polynomial.polynomial.terms) > MAX_NEWTON_TERMS:
            raise ValueError(
                f"Newton polytope requests are limited to {MAX_NEWTON_TERMS} terms"
            )
        return self


class WeightProfileRequest(StrictModel):
    """Request the weight profile of a polynomial."""

    polynomial: RationalPolynomial = Field(
        description="The canonical sparse polynomial whose weight profile is requested."
    )
    weight: tuple[int, ...] = Field(min_length=1, max_length=MAX_WEIGHT_COMPONENTS)

    @model_validator(mode="after")
    def require_matching_dimensions(self) -> Self:
        if len(self.weight) != len(self.polynomial.variables):
            raise ValueError("weight vector length must match variable count")
        # The empty support has no minimum weight; admit only polynomials
        # whose weight profile exists.
        if not self.polynomial.polynomial.terms:
            raise ValueError(
                "the zero polynomial has no weight profile; supply a nonzero polynomial"
            )
        return self


class InitialFormRequest(StrictModel):
    """Request the initial form of a polynomial."""

    polynomial: RationalPolynomial = Field(
        description="The canonical sparse polynomial whose initial form is requested."
    )
    weight: tuple[int, ...] = Field(min_length=1, max_length=MAX_WEIGHT_COMPONENTS)

    @model_validator(mode="after")
    def require_matching_dimensions(self) -> Self:
        if len(self.weight) != len(self.polynomial.variables):
            raise ValueError("weight vector length must match variable count")
        if not self.polynomial.polynomial.terms:
            raise ValueError(
                "the zero polynomial has no initial form; supply a nonzero polynomial"
            )
        return self


__all__ = [
    "InitialFormRequest",
    "NewtonPolytopeRequest",
    "SupportRequest",
    "WeightProfileRequest",
]
