"""Wire-facing finite-coset crossed-product operations."""

from pydantic import ValidationError

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


def _verify_product_result(result: CrossedProductMultiplyResult) -> bool:
    """Verify an independently supplied product in the admitted envelope."""

    try:
        CrossedProductMultiplyRequest(left=result.left, right=result.right)
    except ValidationError:
        return False
    return result.product == multiply(result.left, result.right)


__all__ = ["compute_product"]
