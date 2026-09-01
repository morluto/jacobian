"""Typed declarations for the product representation profile operation."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.additive.product_representation._models import (
    ProductRepresentationRequest,
    ProductRepresentationResult,
)
from jacobian.math.combinatorics.additive.product_representation.operations import (
    compute_product_representation_profile,
)


def _compute(request: ProductRepresentationRequest) -> ProductRepresentationResult:
    return compute_product_representation_profile(request.left, request.right)


TOOLS: MathTools = (
    MathTool(
        operation_id="multiplicative.product_representation_profile.compute",
        title="Compute exact product-set representation profiles",
        description=(
            "Given finite integer sets A and B, return the complete exact product "
            "representation function r_{A*B}(x) = |{(a,b) in A x B : a*b=x}| on "
            "its sorted integer support."
        ),
        request_type=ProductRepresentationRequest,
        result_type=ProductRepresentationResult,
        run=_compute,
        tags=("multiplicative", "combinatorics", "exact"),
        examples=(
            OperationExample(
                name="small_sets",
                description="Product representation of {1,2} x {3,4}.",
                input={
                    "left": {"elements": ["1", "2"]},
                    "right": {"elements": ["3", "4"]},
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
