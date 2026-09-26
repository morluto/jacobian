"""Canonical ZZ quadratic forms and their explicit coefficient inclusion in QQ."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import DecimalIntegerEncoding
from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel
from jacobian.math.number_theory.quadratic_forms.general.values import (
    MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
    RationalQuadraticForm,
)

MAX_INTEGRAL_QUADRATIC_FORM_AXIS = 128
MAX_INTEGRAL_QUADRATIC_FORM_TERMS = 2_048
IntegralCoefficient = Annotated[
    int, DecimalIntegerEncoding(max_digits=MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS)
]


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"quadratic_form.integral.{reason}", message)


def _digits(value: int) -> int:
    return len(str(abs(value)))


class IntegralQuadraticCrossTerm(StrictModel):
    """A nonzero integral coefficient of ``x_left*x_right``, with ``left < right``."""

    left: int = Field(ge=0)
    right: int = Field(ge=0)
    coefficient: IntegralCoefficient

    @model_validator(mode="after")
    def require_order_and_bound(self) -> Self:
        if self.left >= self.right:
            raise _error("cross_term_order", "cross terms require left < right")
        if self.coefficient == 0:
            raise _error("zero_cross_term", "zero cross terms must be omitted")
        if _digits(self.coefficient) > MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS:
            raise _error(
                "coefficient_bound",
                "integral form coefficients exceed the 256-digit bound",
            )
        return self


class IntegralQuadraticForm(StrictModel):
    """A canonical homogeneous polynomial ``Q=Σaᵢxᵢ²+Σcᵢⱼxᵢxⱼ`` over ZZ.

    Diagonal and cross coefficients are the polynomial coefficients themselves.
    The ordered axis is part of the value. Its empty axis is the unique form on
    the zero module; the zero form on any axis has all coefficients zero.
    """

    domain: Literal["ZZ"] = "ZZ"
    axis: tuple[OpaqueLabel, ...] = Field(max_length=MAX_INTEGRAL_QUADRATIC_FORM_AXIS)
    diagonal_coefficients: tuple[IntegralCoefficient, ...] = Field(
        max_length=MAX_INTEGRAL_QUADRATIC_FORM_AXIS
    )
    cross_terms: tuple[IntegralQuadraticCrossTerm, ...] = Field(
        default=(), max_length=MAX_INTEGRAL_QUADRATIC_FORM_TERMS
    )

    @model_validator(mode="after")
    def require_canonical_polynomial(self) -> Self:
        if len(set(self.axis)) != len(self.axis):
            raise _error("axis_unique", "coordinate labels must be unique")
        if len(self.diagonal_coefficients) != len(self.axis):
            raise _error("diagonal_length", "diagonal coefficients must match the axis")
        if len(self.diagonal_coefficients) + len(self.cross_terms) > (
            MAX_INTEGRAL_QUADRATIC_FORM_TERMS
        ):
            raise _error("term_bound", "integral form support exceeds its bound")
        if any(
            _digits(value) > MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS
            for value in self.diagonal_coefficients
        ):
            raise _error(
                "coefficient_bound",
                "integral form coefficients exceed the 256-digit bound",
            )
        positions = tuple((term.left, term.right) for term in self.cross_terms)
        if any(right >= len(self.axis) for _, right in positions):
            raise _error("cross_term_range", "cross-term indices must lie on the axis")
        if positions != tuple(sorted(set(positions))):
            raise _error("cross_terms_order", "cross terms must be unique and ordered")
        return self


class IntegralQuadraticFormInclusionRequest(StrictModel):
    """Apply the canonical coefficient inclusion ZZ → QQ to one form."""

    form: IntegralQuadraticForm


class IntegralQuadraticFormInclusion(StrictModel):
    """Source-bound polynomial transported along the explicit coefficient map ZZ → QQ."""

    map: Literal["ZZ_TO_QQ_COEFFICIENT_INCLUSION"] = "ZZ_TO_QQ_COEFFICIENT_INCLUSION"
    source: IntegralQuadraticForm
    target: RationalQuadraticForm


__all__ = [
    "MAX_INTEGRAL_QUADRATIC_FORM_AXIS",
    "MAX_INTEGRAL_QUADRATIC_FORM_TERMS",
    "IntegralQuadraticCrossTerm",
    "IntegralQuadraticForm",
    "IntegralQuadraticFormInclusion",
    "IntegralQuadraticFormInclusionRequest",
]
