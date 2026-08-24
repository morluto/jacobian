"""Wire-facing finite-coset crossed-product operations."""

from jacobian.math.crossed_products._models import (
    CrossedProductMultiplyRequest,
    CrossedProductMultiplyResult,
    _computed_result,
)
from jacobian.math.crossed_products.operations import multiply


def compute_product(
    request: CrossedProductMultiplyRequest,
) -> CrossedProductMultiplyResult:
    """Multiply two elements and retain their complete ordered sources."""

    return _computed_result(
        request.left,
        request.right,
        multiply(request.left, request.right),
    )


__all__ = ["compute_product"]
