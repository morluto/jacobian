"""Typed wire contracts for number field operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.polynomials.values import PolynomialVariable

MAX_NUMBER_FIELD_COEFFICIENT_DIGITS = 256


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"number_field.{reason}", message)


class NumberFieldRequest(StrictModel):
    """A number field Q(alpha) defined by a minimal polynomial."""

    coefficients_descending: tuple[str, ...] = Field(min_length=2, max_length=32)
    variable: PolynomialVariable

    @model_validator(mode="after")
    def require_bounded_monic_integer_polynomial(self) -> Self:
        # Bound digit conversion before SymPy sees an arbitrary decimal
        # spelling.  Degree and coefficient height jointly bound the parser
        # input and every polynomial construction that follows.
        if any(
            len(coefficient.lstrip("-")) > MAX_NUMBER_FIELD_COEFFICIENT_DIGITS
            for coefficient in self.coefficients_descending
        ):
            raise _validation_error(
                "coefficient_digits",
                "number-field coefficients may contain at most "
                f"{MAX_NUMBER_FIELD_COEFFICIENT_DIGITS} decimal digits",
            )
        try:
            coefficients = tuple(
                parse_canonical_integer(value) for value in self.coefficients_descending
            )
        except ValueError as exc:
            raise _validation_error(
                "coefficient_syntax",
                "number-field coefficients must be canonical integers",
            ) from exc
        if coefficients[0] != 1:
            raise _validation_error(
                "not_monic", "number-field polynomial must be monic"
            )
        return self


class NumberFieldDiscriminantResult(StrictModel):
    status: Literal["COMPLETE", "UNKNOWN"] = "COMPLETE"
    discriminant: str | None = None
    detail: str | None = Field(default=None, max_length=1_024)

    @model_validator(mode="after")
    def bind_outcome(self) -> Self:
        if self.status == "COMPLETE" and self.discriminant is None:
            raise _validation_error(
                "complete_discriminant_requires_value",
                "a complete number-field discriminant requires its exact value",
            )
        if self.status == "UNKNOWN" and (
            self.discriminant is not None or self.detail is None
        ):
            raise _validation_error(
                "unknown_discriminant_shape",
                "an unknown number-field computation requires detail and no value",
            )
        return self
