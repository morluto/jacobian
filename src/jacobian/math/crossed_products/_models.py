"""Typed wire contracts for finite-coset crossed-product multiplication."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.crossed_products.operations import (
    _require_multiplication_budget,
    multiply,
)
from jacobian.math.crossed_products.values import FiniteCosetCrossedProductElement


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"crossed_product.{reason}", message)


class CrossedProductMultiplyRequest(StrictModel):
    """Two elements in the same fully validated finite-coset crossed product."""

    left: FiniteCosetCrossedProductElement
    right: FiniteCosetCrossedProductElement

    @model_validator(mode="after")
    def require_bounded_product(self) -> Self:
        try:
            _require_multiplication_budget(self.left, self.right)
        except ValueError as exc:
            raise _validation_error("product_budget_exceeded", str(exc)) from exc
        return self


class CrossedProductMultiplyResult(StrictModel):
    """An exact sparse product bound to both ordered source operands."""

    left: FiniteCosetCrossedProductElement
    right: FiniteCosetCrossedProductElement
    product: FiniteCosetCrossedProductElement

    @model_validator(mode="after")
    def bind_product_to_sources(self) -> Self:
        if self.product.presentation != self.left.presentation:
            raise _validation_error(
                "presentation_mismatch", "product must retain the operand presentation"
            )
        expected = multiply(self.left, self.right)
        if self.product != expected:
            raise _validation_error(
                "product_invariant",
                "product must equal exact replay from the retained operands",
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
