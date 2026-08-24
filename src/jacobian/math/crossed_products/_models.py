"""Typed wire contracts for finite-coset crossed-product multiplication."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.crossed_products.operations import (
    _require_multiplication_budget,
    multiply,
)
from jacobian.math.crossed_products.values import FiniteCosetCrossedProductElement


class CrossedProductMultiplyRequest(StrictModel):
    """Two elements in the same fully validated finite-coset crossed product."""

    left: FiniteCosetCrossedProductElement
    right: FiniteCosetCrossedProductElement

    @model_validator(mode="after")
    def require_bounded_product(self) -> Self:
        _require_multiplication_budget(self.left, self.right)
        return self


class CrossedProductMultiplyResult(StrictModel):
    """An exact sparse product bound to both ordered source operands."""

    left: FiniteCosetCrossedProductElement
    right: FiniteCosetCrossedProductElement
    product: FiniteCosetCrossedProductElement

    @model_validator(mode="after")
    def bind_product_to_sources(self) -> Self:
        if self.product.presentation != self.left.presentation:
            raise ValueError("product must retain the operand presentation")
        expected = multiply(self.left, self.right)
        if self.product != expected:
            raise ValueError(
                "product must equal exact replay from the retained operands"
            )
        return self


def _computed_result(
    left: FiniteCosetCrossedProductElement,
    right: FiniteCosetCrossedProductElement,
    product: FiniteCosetCrossedProductElement,
) -> CrossedProductMultiplyResult:
    """Bind one product freshly computed by the owning kernel.

    Direct construction from the producing kernel skips result replay so the
    admitted scalar-work budget covers all multiplication work; independently
    supplied results always validate through ``bind_product_to_sources``.
    """

    return CrossedProductMultiplyResult.model_construct(
        left=left,
        right=right,
        product=product,
    )


__all__ = ["CrossedProductMultiplyRequest", "CrossedProductMultiplyResult"]
