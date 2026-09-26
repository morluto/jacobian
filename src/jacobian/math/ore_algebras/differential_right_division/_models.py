"""Typed request and reconstruction identity for differential right division."""

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.ore_algebras._models import DifferentialOreOperator


class DifferentialRightDivisionRequest(StrictModel):
    """Divide A on the right by monic B, with A = Q B + R."""

    dividend: DifferentialOreOperator
    divisor: DifferentialOreOperator

    @model_validator(mode="after")
    def require_nonzero_divisor(self) -> Self:
        if not self.divisor.terms:
            raise PydanticCustomError(
                "ore_algebra.differential_right_division_zero_divisor",
                "right division requires a nonzero divisor",
            )
        return self


class DifferentialRightDivisionResult(StrictModel):
    """Exact unique quotient and lower-order remainder satisfying A=Q*B+R."""

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
