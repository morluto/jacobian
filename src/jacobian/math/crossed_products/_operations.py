"""Wire-facing finite-coset crossed-product operations."""

from jacobian.math.crossed_products._models import (
    CrossedProductMultiplyRequest,
    CrossedProductMultiplyResult,
)
from jacobian.math.crossed_products.operations import multiply


def compute_product(
    request: CrossedProductMultiplyRequest,
) -> CrossedProductMultiplyResult:
    """Multiply two elements and retain their complete ordered sources."""

    return CrossedProductMultiplyResult(
        left=request.left,
        right=request.right,
        product=multiply(request.left, request.right),
    )


__all__ = ["compute_product"]
