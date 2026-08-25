"""Typed wire contracts for polynomial support geometry operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.polynomial_support_geometry.values import (
    MAX_NEWTON_TERMS,
    MAX_WEIGHT_COMPONENTS,
)
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polynomial_support_geometry.{reason}", message)


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
            raise _validation_error(
                "newton_term_count_exceeded",
                (f"Newton polytope requests are limited to {MAX_NEWTON_TERMS} terms"),
            )
        return self


# Derived weights are sums of weight*exponent products; capping each
# component at 2^31 keeps every derived integer inside the interoperable
# JSON range (len(weight) <= 8, exponents <= 32768).
MAX_WEIGHT_COMPONENT_MAGNITUDE = 2**31


def _require_transportable_weight(
    weight: tuple[int, ...], variables: tuple[PolynomialVariable, ...]
) -> None:
    if len(weight) != len(variables):
        raise _validation_error(
            "weight_dimension_mismatch",
            "weight vector length must match variable count",
        )
    for component in weight:
        if abs(component) > MAX_WEIGHT_COMPONENT_MAGNITUDE:
            raise _validation_error(
                "weight_component_out_of_range",
                "weight components exceed the transportable integer range "
                f"(max {MAX_WEIGHT_COMPONENT_MAGNITUDE})",
            )


class WeightProfileRequest(StrictModel):
    """Request the weight profile of a polynomial."""

    polynomial: RationalPolynomial = Field(
        description=(
            "The canonical sparse polynomial whose weight profile is "
            "requested: a nonzero polynomial with at most 1024 terms and "
            "coefficient numerator/denominator components of at most 512 "
            "digits."
        )
    )
    weight: tuple[int, ...] = Field(
        min_length=1,
        max_length=MAX_WEIGHT_COMPONENTS,
        description=(
            "One integer per variable of the retained polynomial; each "
            "component is bounded in magnitude by 2**31 so derived "
            "weights stay inside the interoperable JSON integer range."
        ),
    )

    @model_validator(mode="after")
    def require_matching_dimensions(self) -> Self:
        _require_transportable_weight(self.weight, self.polynomial.variables)
        # The empty support has no minimum weight; admit only polynomials
        # whose weight profile exists.
        if not self.polynomial.polynomial.terms:
            raise _validation_error(
                "zero_weight_profile",
                (
                    "the zero polynomial has no weight profile; supply a nonzero polynomial"
                ),
            )
        # A degenerate weight repeats every exponent in the minimizing list
        # and again inside the single layer; cap the serialized profile so
        # it stays well inside the canonical output envelope.
        terms = self.polynomial.polynomial.terms
        if len(terms) > 1024:
            raise _validation_error(
                "weight_profile_term_count_exceeded",
                "weight-profile requests are limited to 1024 terms",
            )
        for term in terms:
            for component in (term.coefficient.num, term.coefficient.den):
                if len(component.lstrip("-")) > 512:
                    raise _validation_error(
                        "weight_profile_coefficient_too_large",
                        ("weight-profile coefficients are limited to 512 digits"),
                    )
        return self


class InitialFormRequest(StrictModel):
    """Request the initial form of a polynomial."""

    polynomial: RationalPolynomial = Field(
        description=(
            "The canonical sparse polynomial whose initial form is "
            "requested: a nonzero polynomial with at most 1024 terms and "
            "coefficient numerator/denominator components of at most 512 "
            "digits."
        )
    )
    weight: tuple[int, ...] = Field(
        min_length=1,
        max_length=MAX_WEIGHT_COMPONENTS,
        description=(
            "One integer per variable of the retained polynomial; each "
            "component is bounded in magnitude by 2**31 so derived "
            "weights stay inside the interoperable JSON integer range."
        ),
    )

    @model_validator(mode="after")
    def require_matching_dimensions(self) -> Self:
        _require_transportable_weight(self.weight, self.polynomial.variables)
        if not self.polynomial.polynomial.terms:
            raise _validation_error(
                "zero_initial_form",
                (
                    "the zero polynomial has no initial form; supply a nonzero polynomial"
                ),
            )
        # A degenerate weight can make every term minimal, serializing the
        # whole polynomial twice (source + face); admit only sources whose
        # doubled serialization stays comfortably inside the envelope.
        terms = self.polynomial.polynomial.terms
        if len(terms) > 1024:
            raise _validation_error(
                "initial_form_term_count_exceeded",
                "initial-form requests are limited to 1024 terms",
            )
        for term in terms:
            for component in (term.coefficient.num, term.coefficient.den):
                if len(component.lstrip("-")) > 512:
                    raise _validation_error(
                        "initial_form_coefficient_too_large",
                        ("initial-form coefficients are limited to 512 digits"),
                    )
        return self


__all__ = [
    "InitialFormRequest",
    "NewtonPolytopeRequest",
    "SupportRequest",
    "WeightProfileRequest",
]
