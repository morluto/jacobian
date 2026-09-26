"""Typed values for the recurrence-to-OGF equation transform."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.number_theory.sequences.core._models import FiniteRationalSequence
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


class RecurrenceOGFEquationRequest(StrictModel):
    """Transform a polynomial recurrence relation into its OGF equation."""

    recurrence: ShiftOreOperator = Field(
        description="A nonzero shift recurrence with polynomial coefficients in QQ[n]."
    )
    initial_coefficients: FiniteRationalSequence = Field(
        description=(
            "Provide exactly a_0 through a_(r-1), where r is the largest "
            "shift exponent in recurrence."
        ),
        json_schema_extra={
            "x-cross-field-rule": "length(values) equals recurrence.order",
            "examples": [{"values": [1, 1]}],
        },
    )

    @model_validator(mode="after")
    def require_boundary_coefficients(self) -> Self:
        order = self.recurrence.order
        if order < 0:
            raise PydanticCustomError(
                "ore_algebra.ogf_zero_recurrence",
                "the zero operator has no recurrence equation",
            )
        if len(self.initial_coefficients.values) != order:
            raise PydanticCustomError(
                "ore_algebra.ogf_initial_coefficients",
                "initial_coefficients must provide a_0 through a_(r-1), where r is the largest shift exponent",
            )
        if any(
            not _is_polynomial_coefficient(term.coefficient)
            for term in self.recurrence.terms
        ):
            raise PydanticCustomError(
                "ore_algebra.ogf_polynomial_coefficients",
                "ordinary generating-function conversion requires polynomial coefficients in QQ[n]",
            )
        return self


class RecurrenceOGFEquation(StrictModel):
    """The exact differential equation L(F)=B induced by one recurrence."""

    recurrence: ShiftOreOperator
    initial_coefficients: FiniteRationalSequence
    differential_operator: DifferentialOreOperator
    forcing: RationalFunction

    @model_validator(mode="after")
    def require_forcing_axis(self) -> Self:
        if self.forcing.variables != ("x",):
            raise PydanticCustomError(
                "ore_algebra.ogf_forcing_axis",
                "the forcing polynomial must use QQ(x)",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        recurrence: ShiftOreOperator,
        initial_coefficients: FiniteRationalSequence,
        differential_operator: DifferentialOreOperator,
        forcing: RationalFunction,
    ) -> Self:
        return cls.model_construct(
            recurrence=recurrence,
            initial_coefficients=initial_coefficients,
            differential_operator=differential_operator,
            forcing=forcing,
        )
