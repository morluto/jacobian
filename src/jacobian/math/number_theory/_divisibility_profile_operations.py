"""Exact kernels for gcd-normalized quotient and product-divisibility profiles."""

from __future__ import annotations

import math

from jacobian.math.number_theory._divisibility_profile_models import (
    GcdQuotientProfileRequest,
    GcdQuotientProfileResult,
    ProductDivisibilityProfileRequest,
    ProductDivisibilityProfileResult,
)


def compute_gcd_quotient_profile(
    request: GcdQuotientProfileRequest,
) -> GcdQuotientProfileResult:
    """For each pair (a, b), compute gcd(a,b) / max(|a|,|b|) as a normalized quotient.

    The result is a matrix where entry [i][j] = gcd(elements[i], elements[j]).
    """
    elements = [int(e) for e in request.elements]
    n = len(elements)
    quotients = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                quotients[i][j] = abs(elements[i])
            else:
                quotients[i][j] = math.gcd(abs(elements[i]), abs(elements[j]))
    return GcdQuotientProfileResult(
        elements=request.elements,
        quotients=quotients,
    )


def compute_product_divisibility_profile(
    request: ProductDivisibilityProfileRequest,
) -> ProductDivisibilityProfileResult:
    """For each pair (a, b), determine if a*b divides the product of all elements."""

    elements = [int(e) for e in request.elements]
    n = len(elements)
    total_product = 1
    for e in elements:
        total_product *= abs(e) if e != 0 else 1

    matrix = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            a = elements[i]
            b = elements[j]
            if a == 0 or b == 0:
                matrix[i][j] = True
            else:
                # Check if a divides b
                matrix[i][j] = (b % a == 0) if a != 0 else False

    return ProductDivisibilityProfileResult(
        elements=request.elements,
        divisibility_matrix=matrix,
    )


__all__ = [
    "compute_gcd_quotient_profile",
    "compute_product_divisibility_profile",
]
