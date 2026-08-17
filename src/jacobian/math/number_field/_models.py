"""Typed wire contracts for number field operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer


class NumberFieldRequest(StrictModel):
    """A number field Q(alpha) defined by a minimal polynomial."""

    coefficients_descending: tuple[str, ...] = Field(min_length=2, max_length=32)
    variable: str = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def require_monic_irreducible_integer_polynomial(self) -> Self:
        import sympy

        variable = sympy.Symbol(self.variable)
        coefficients = tuple(
            sympy.Rational(parse_canonical_integer(value))
            for value in self.coefficients_descending
        )
        if any(value.q != 1 for value in coefficients):
            raise ValueError("number-field coefficients must be integers")
        polynomial = sympy.Poly.from_list(coefficients, gens=variable, domain=sympy.ZZ)
        if not polynomial.is_monic:
            raise ValueError("number-field polynomial must be monic")
        if not polynomial.is_irreducible:
            raise ValueError("number-field polynomial must be irreducible over QQ")
        return self


class NumberFieldDiscriminantResult(StrictModel):
    discriminant: str
    method: Literal["SYMPY_NUMBER_FIELD"] = "SYMPY_NUMBER_FIELD"
