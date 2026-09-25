"""Typed request and reconstruction identity for differential left division."""

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.ore_algebras._models import DifferentialOreOperator


class DifferentialLeftDivisionRequest(StrictModel):
    """Divide A on the left by monic B, with A = B*Q + R."""

    dividend: DifferentialOreOperator
    divisor: DifferentialOreOperator

    @model_validator(mode="after")
    def require_nonzero_divisor(self) -> Self:
        if not self.divisor.terms:
            raise PydanticCustomError(
                "ore_algebra.differential_left_division_zero_divisor",
                "left division requires a nonzero divisor",
            )
        return self


class DifferentialLeftDivisionResult(StrictModel):
    """Exact unique quotient and lower-order remainder satisfying A=B*Q+R."""

    dividend: DifferentialOreOperator
    divisor: DifferentialOreOperator
    quotient: DifferentialOreOperator
    remainder: DifferentialOreOperator

    @classmethod
    def _from_kernel(
        cls,
        dividend: DifferentialOreOperator,
        divisor: DifferentialOreOperator,
        quotient: DifferentialOreOperator,
        remainder: DifferentialOreOperator,
    ) -> Self:
        return cls.model_construct(
            dividend=dividend,
            divisor=divisor,
            quotient=quotient,
            remainder=remainder,
        )
