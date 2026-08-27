"""Exact kernels for gcd-normalized quotient and product-divisibility profiles."""

from __future__ import annotations

import math
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.number_theory._divisibility_profile_models import (
    GcdQuotientProfileRequest,
    GcdQuotientProfileResult,
    ProductDivisibilityProfileRequest,
    ProductDivisibilityProfileResult,
)


def compute_gcd_quotient_profile(
    request: GcdQuotientProfileRequest,
) -> GcdQuotientProfileResult:
    """For each pair, compute the normalized ratio gcd(a,b)/max(|a|,|b|)."""
    elements = [int(e) for e in request.elements]
    n = len(elements)
    quotients: list[list[CanonicalRational]] = []
    for i in range(n):
        row = []
        for j in range(n):
            numerator = math.gcd(abs(elements[i]), abs(elements[j]))
            denominator = max(abs(elements[i]), abs(elements[j]))
            row.append(
                CanonicalRational.from_fraction(Fraction(numerator, denominator))
            )
        quotients.append(row)
    return GcdQuotientProfileResult(
        elements=request.elements,
        quotients=tuple(tuple(row) for row in quotients),
    )


def compute_product_divisibility_profile(
    request: ProductDivisibilityProfileRequest,
) -> ProductDivisibilityProfileResult:
    """For each pair (a, b), determine if a*b divides the product of all elements."""

    elements = [int(e) for e in request.elements]
    n = len(elements)
    total_product = math.prod(elements)

    matrix = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            a = elements[i]
            b = elements[j]
            matrix[i][j] = total_product % (a * b) == 0

    return ProductDivisibilityProfileResult(
        elements=request.elements,
        divisibility_matrix=tuple(tuple(row) for row in matrix),
    )


__all__ = [
    "compute_gcd_quotient_profile",
    "compute_product_divisibility_profile",
]
