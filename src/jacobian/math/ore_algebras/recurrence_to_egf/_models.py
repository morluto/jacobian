"""Typed values for the recurrence-to-EGF differential-equation transform."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.ore_algebras._models import (
    DifferentialOreOperator,
    ShiftOreOperator,
)
from jacobian.math.polynomials.values import RationalFunction


def _is_polynomial_coefficient(value: RationalFunction) -> bool:
    return (
        value.variables == ("n",)
        and len(value.denominator.terms) == 1
        and value.denominator.terms[0].exponents == (0,)
        and value.denominator.terms[0].coefficient.as_fraction() == 1
    )


class RecurrenceEGFEquationRequest(StrictModel):
    """A recurrence relation whose EGF differential operator is requested."""

    recurrence: ShiftOreOperator

    @model_validator(mode="after")
    def require_polynomial_recurrence(self) -> Self:
        if not self.recurrence.terms:
            raise PydanticCustomError(
                "ore_algebra.egf_zero_recurrence",
                "the zero operator has no recurrence equation",
            )
        if any(
            not _is_polynomial_coefficient(term.coefficient)
            for term in self.recurrence.terms
        ):
            raise PydanticCustomError(
                "ore_algebra.egf_polynomial_coefficients",
                "exponential generating-function conversion requires polynomial coefficients in QQ[n]",
            )
        return self


class RecurrenceEGFEquation(StrictModel):
    """A recurrence and its exact homogeneous differential operator for an EGF."""

    recurrence: ShiftOreOperator
    differential_operator: DifferentialOreOperator

    @classmethod
    def _from_kernel(
        cls,
        recurrence: ShiftOreOperator,
        differential_operator: DifferentialOreOperator,
    ) -> Self:
        return cls.model_construct(
            recurrence=recurrence,
            differential_operator=differential_operator,
        )
