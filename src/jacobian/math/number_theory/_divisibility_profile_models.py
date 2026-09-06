"""Typed contracts for gcd-normalized quotient and product-divisibility profiles."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel
from jacobian.math.number_theory._models import BoundedInteger

MAX_FAMILY_SIZE: int = 100


class GcdQuotientProfileRequest(StrictModel):
    """A finite family of positive integers for gcd-normalized profiling."""

    elements: tuple[BoundedInteger, ...] = Field(
        max_length=MAX_FAMILY_SIZE,
        description=(
            "Finite family of at most 100 positive integers; every element is "
            "strictly greater than zero. Zero and negative integers are invalid."
        ),
    )


class GcdQuotientProfileResult(StrictModel):
    """Complete gcd-normalized quotient profile."""

    elements: tuple[ExactInteger, ...]
    quotients: tuple[tuple[CanonicalRational, ...], ...]

    @model_validator(mode="after")
    def require_square_profile(self) -> Self:
        size = len(self.elements)
        if len(self.quotients) != size or any(
            len(row) != size for row in self.quotients
        ):
            raise PydanticCustomError(
                "number_theory.gcd_quotient_profile_must_be_square",
                "gcd-quotient profile must have one square matrix entry per pair",
            )
        return self


class ProductDivisibilityProfileRequest(StrictModel):
    """A finite family of positive integers for product-divisibility profiling."""

    elements: tuple[BoundedInteger, ...] = Field(
        max_length=MAX_FAMILY_SIZE,
        description=(
            "Finite family of at most 100 positive integers; every element is "
            "strictly greater than zero. Zero and negative integers are invalid."
        ),
    )


class ProductDivisibilityProfileResult(StrictModel):
    """Complete product-divisibility relation profile."""

    elements: tuple[ExactInteger, ...]
    divisibility_matrix: tuple[tuple[bool, ...], ...]

    @model_validator(mode="after")
    def require_square_profile(self) -> Self:
        size = len(self.elements)
        if len(self.divisibility_matrix) != size or any(
            len(row) != size for row in self.divisibility_matrix
        ):
            raise PydanticCustomError(
                "number_theory.product_divisibility_profile_must_be_square",
                "product-divisibility profile must have one matrix entry per pair",
            )
        return self


__all__ = [
    "MAX_FAMILY_SIZE",
    "GcdQuotientProfileRequest",
    "GcdQuotientProfileResult",
    "ProductDivisibilityProfileRequest",
    "ProductDivisibilityProfileResult",
]
