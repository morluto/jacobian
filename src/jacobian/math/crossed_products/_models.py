"""Typed wire contracts for finite-coset crossed-product multiplication."""

from __future__ import annotations

from jacobian._models import StrictModel
from jacobian.math.crossed_products.values import FiniteCosetCrossedProductElement


class CrossedProductMultiplyRequest(StrictModel):
    """Two elements in the same fully validated finite-coset crossed product."""

    left: FiniteCosetCrossedProductElement
    right: FiniteCosetCrossedProductElement


class CrossedProductMultiplyResult(StrictModel):
    """An exact sparse product bound to both ordered source operands.

    Kernel output is built with the trusted factory below rather than
    re-entering multiplication while parsing this value.
    """

    left: FiniteCosetCrossedProductElement
    right: FiniteCosetCrossedProductElement
    product: FiniteCosetCrossedProductElement


def _computed_result(
    left: FiniteCosetCrossedProductElement,
    right: FiniteCosetCrossedProductElement,
    product: FiniteCosetCrossedProductElement,
) -> CrossedProductMultiplyResult:
    """Bind one product freshly computed by the owning kernel."""

    return CrossedProductMultiplyResult.model_construct(
        left=left,
        right=right,
        product=product,
    )


__all__ = ["CrossedProductMultiplyRequest", "CrossedProductMultiplyResult"]
