"""Typed first-order differential-operator common-multiple contract."""

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.ore_algebras._models import DifferentialOreOperator


class FirstOrderLCLMRequest(StrictModel):
    """Two nonzero first-order operators over QQ(x)."""

    left: DifferentialOreOperator
    right: DifferentialOreOperator

    @model_validator(mode="after")
    def require_first_order_operators(self) -> Self:
        if self.left.order != 1 or self.right.order != 1:
            raise PydanticCustomError(
                "ore_algebra.first_order_lclm_inputs",
                "both operators must have differential order one",
            )
        return self


class FirstOrderLCLMResult(StrictModel):
    """Common left multiple and its exact left multipliers."""

    left: DifferentialOreOperator
    right: DifferentialOreOperator
    left_multiplier: DifferentialOreOperator
    right_multiplier: DifferentialOreOperator
    common_left_multiple: DifferentialOreOperator

    @classmethod
    def _from_kernel(
        cls,
        left: DifferentialOreOperator,
        right: DifferentialOreOperator,
        left_multiplier: DifferentialOreOperator,
        right_multiplier: DifferentialOreOperator,
        common_left_multiple: DifferentialOreOperator,
    ) -> Self:
        return cls.model_construct(
            left=left,
            right=right,
            left_multiplier=left_multiplier,
            right_multiplier=right_multiplier,
            common_left_multiple=common_left_multiple,
        )
