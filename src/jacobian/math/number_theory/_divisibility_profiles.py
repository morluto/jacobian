"""Declarations for gcd-quotient and product-divisibility profiles."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory._divisibility_profile_models import (
    GcdQuotientProfileRequest,
    GcdQuotientProfileResult,
    ProductDivisibilityProfileRequest,
    ProductDivisibilityProfileResult,
)
from jacobian.math.number_theory.operations import (
    gcd_quotient_profile,
    product_divisibility_profile,
)


def compute_gcd_quotient_profile(
    request: GcdQuotientProfileRequest,
) -> GcdQuotientProfileResult:
    """Project a wire request into the canonical gcd profile operation."""
    return gcd_quotient_profile(tuple(request.elements))


def compute_product_divisibility_profile(
    request: ProductDivisibilityProfileRequest,
) -> ProductDivisibilityProfileResult:
    """Project a wire request into the canonical divisibility operation."""
    return product_divisibility_profile(tuple(request.elements))


DIVISIBILITY_PROFILE_OPERATIONS = (
    MathTool(
        operation_id="number_theory.gcd_quotient.profile.compute",
        title="Compute gcd-normalized quotient profile on a finite integer family",
        description="For each pair (a, b) in a finite family of positive integers, compute "
        "the normalized ratio gcd(a, b) / max(|a|, |b|). Zero and negative "
        "integers are outside the request domain.",
        request_type=GcdQuotientProfileRequest,
        result_type=GcdQuotientProfileResult,
        run=compute_gcd_quotient_profile,
        tags=("number-theory", "divisibility", "profile"),
        examples=(
            OperationExample(
                name="gcd_quotient_basic",
                description="Compute gcd quotients for {6, 10, 15}.",
                input={"elements": ["6", "10", "15"]},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.product_divisibility.profile.compute",
        title="Compute product-divisibility profile on a finite integer family",
        description="For each pair (a, b) in a finite family of positive integers, determine "
        "whether a*b divides the product of the family. Zero and negative "
        "integers are outside the request domain.",
        request_type=ProductDivisibilityProfileRequest,
        result_type=ProductDivisibilityProfileResult,
        run=compute_product_divisibility_profile,
        tags=("number-theory", "divisibility", "profile"),
        examples=(
            OperationExample(
                name="product_div_basic",
                description="Compute divisibility for {2, 6, 12}.",
                input={"elements": ["2", "6", "12"]},
            ),
        ),
    ),
)

__all__ = ["DIVISIBILITY_PROFILE_OPERATIONS"]
