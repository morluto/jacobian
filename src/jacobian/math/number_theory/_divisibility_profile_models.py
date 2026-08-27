"""Typed contracts for gcd-normalized quotient and product-divisibility profiles."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.number_theory._models import BoundedInteger

MAX_FAMILY_SIZE: int = 100


class GcdQuotientProfileRequest(StrictModel):
    """A finite integer family for gcd-normalized quotient profiling."""

    elements: tuple[BoundedInteger, ...] = Field(max_length=MAX_FAMILY_SIZE)


class GcdQuotientProfileResult(StrictModel):
    """Complete gcd-normalized quotient profile."""

    elements: tuple[str, ...]
    quotients: list[list[int]]


class ProductDivisibilityProfileRequest(StrictModel):
    """A finite integer family for product-divisibility profiling."""

    elements: tuple[BoundedInteger, ...] = Field(max_length=MAX_FAMILY_SIZE)


class ProductDivisibilityProfileResult(StrictModel):
    """Complete product-divisibility relation profile."""

    elements: tuple[str, ...]
    divisibility_matrix: list[list[bool]]


__all__ = [
    "GcdQuotientProfileRequest",
    "GcdQuotientProfileResult",
    "ProductDivisibilityProfileRequest",
    "ProductDivisibilityProfileResult",
    "MAX_FAMILY_SIZE",
]
