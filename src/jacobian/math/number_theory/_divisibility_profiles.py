"""Declarations for gcd-quotient and product-divisibility profiles."""

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog._examples import example
from jacobian.math.number_theory._divisibility_profile_models import (
    GcdQuotientProfileRequest,
    GcdQuotientProfileResult,
    ProductDivisibilityProfileRequest,
    ProductDivisibilityProfileResult,
)
from jacobian.math.number_theory._support import number_theory_operation
from jacobian.math.number_theory.operations import (
    gcd_quotient_profile,
    product_divisibility_profile,
)


def compute_gcd_quotient_profile(
    request: GcdQuotientProfileRequest,
) -> GcdQuotientProfileResult:
    """Project a wire request into the canonical gcd profile operation."""
    return gcd_quotient_profile(
        tuple(parse_canonical_integer(value) for value in request.elements)
    )


def compute_product_divisibility_profile(
    request: ProductDivisibilityProfileRequest,
) -> ProductDivisibilityProfileResult:
    """Project a wire request into the canonical divisibility operation."""
    return product_divisibility_profile(
        tuple(parse_canonical_integer(value) for value in request.elements)
    )


DIVISIBILITY_PROFILE_OPERATIONS = (
    number_theory_operation(
        "number_theory.gcd_quotient.profile.compute",
        "Compute gcd-normalized quotient profile on a finite integer family",
        "For each pair (a, b) in a finite family of positive integers, compute "
        "the normalized ratio gcd(a, b) / max(|a|, |b|). Zero and negative "
        "integers are outside the request domain.",
        GcdQuotientProfileRequest,
        GcdQuotientProfileResult,
        compute_gcd_quotient_profile,
        "number-theory",
        "divisibility",
        "profile",
        examples=(
            example(
                "gcd_quotient_basic",
                "Compute gcd quotients for {6, 10, 15}.",
                {"elements": ["6", "10", "15"]},
            ),
        ),
    ),
    number_theory_operation(
        "number_theory.product_divisibility.profile.compute",
        "Compute product-divisibility profile on a finite integer family",
        "For each pair (a, b) in a finite family of positive integers, determine "
        "whether a*b divides the product of the family. Zero and negative "
        "integers are outside the request domain.",
        ProductDivisibilityProfileRequest,
        ProductDivisibilityProfileResult,
        compute_product_divisibility_profile,
        "number-theory",
        "divisibility",
        "profile",
        examples=(
            example(
                "product_div_basic",
                "Compute divisibility for {2, 6, 12}.",
                {"elements": ["2", "6", "12"]},
            ),
        ),
    ),
)

__all__ = ["DIVISIBILITY_PROFILE_OPERATIONS"]
