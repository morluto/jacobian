"""Typed wire contracts for polynomial support geometry operations."""

from __future__ import annotations

from pydantic import Field
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.polynomials.support_geometry.values import (
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


# Derived weights are sums of weight*exponent products; capping each
# component at 2^31 keeps every derived integer inside the interoperable
# JSON range (len(weight) <= 8, exponents <= 32768).
MAX_WEIGHT_COMPONENT_MAGNITUDE = 2**31
MAX_WEIGHTED_POLYNOMIAL_TERMS = 1024
MAX_WEIGHTED_COEFFICIENT_DIGITS = 512


def _require_transportable_weight(
    weight: tuple[int, ...], variables: tuple[PolynomialVariable, ...]
) -> None:
    if len(weight) != len(variables):
        raise _validation_error(
            "weight_dimension_mismatch",
            "weight vector length must match variable count",
        )
    for component in weight:
        if type(component) is not int:
            raise _validation_error(
                "weight_component_not_integer",
                "weight components must be exact integers",
            )
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


__all__ = [
    "InitialFormRequest",
    "NewtonPolytopeRequest",
    "SupportRequest",
    "WeightProfileRequest",
]
