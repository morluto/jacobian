"""Typed wire contracts for number field operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.polynomials.values import PolynomialVariable


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"number_field.{reason}", message)


class NumberFieldRequest(StrictModel):
    """A number field Q(alpha) defined by a minimal polynomial."""

    coefficients_descending: tuple[str, ...] = Field(min_length=2, max_length=32)
    variable: PolynomialVariable

    @model_validator(mode="after")
    def require_monic_irreducible_integer_polynomial(self) -> Self:
        import sympy

        variable = sympy.Symbol(self.variable)
        coefficients = tuple(
            sympy.Rational(parse_canonical_integer(value))
            for value in self.coefficients_descending
        )
        if any(value.q != 1 for value in coefficients):
            raise _validation_error(
                "coefficient_domain", "number-field coefficients must be integers"
            )
        polynomial = sympy.Poly.from_list(coefficients, gens=variable, domain=sympy.ZZ)
        if not polynomial.is_monic:
            raise _validation_error(
                "not_monic", "number-field polynomial must be monic"
            )
        if not polynomial.is_irreducible:
            raise _validation_error(
                "not_irreducible", "number-field polynomial must be irreducible over QQ"
            )
        return self


class NumberFieldDiscriminantResult(StrictModel):
    discriminant: str
    method: Literal["SYMPY_NUMBER_FIELD"] = "SYMPY_NUMBER_FIELD"
