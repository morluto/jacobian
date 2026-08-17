"""Formal coefficientwise polynomial arithmetic modulo one integer."""

from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from jacobian._models import StrictModel

_MAX_INTEGER_DIGITS = 256
_MAX_MODULUS = 1_000_000
_MAX_VARIABLES = 6
_MAX_TERMS = 64
_MAX_EXPONENT = 32
_VARIABLE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_INTEGER = re.compile(r"^(?:0|-?[1-9][0-9]*)$")


class ModularPolynomialTerm(StrictModel):
    coefficient: StrictStr = Field(min_length=1, max_length=_MAX_INTEGER_DIGITS + 1)
    exponents: tuple[StrictInt, ...] = Field(min_length=1, max_length=_MAX_VARIABLES)

    @field_validator("coefficient")
    @classmethod
    def require_canonical_integer(cls, value: str) -> str:
        if _INTEGER.fullmatch(value) is None:
            raise ValueError("term coefficient must be a canonical integer")
        return value

    @model_validator(mode="after")
    def require_bounded_exponents(self) -> Self:
        if any(exponent < 0 or exponent > _MAX_EXPONENT for exponent in self.exponents):
            raise ValueError("term exponents must be between 0 and 32")
        return self


class ModularPolynomialIdentityRequest(StrictModel):
    modulus: StrictInt = Field(ge=2, le=_MAX_MODULUS)
    variables: tuple[StrictStr, ...] = Field(min_length=1, max_length=_MAX_VARIABLES)
    left: tuple[ModularPolynomialTerm, ...] = Field(default=(), max_length=_MAX_TERMS)
    right: tuple[ModularPolynomialTerm, ...] = Field(default=(), max_length=_MAX_TERMS)

    @model_validator(mode="after")
    def require_scope(self) -> Self:
        if len(set(self.variables)) != len(self.variables) or any(
            _VARIABLE.fullmatch(name) is None for name in self.variables
        ):
            raise ValueError("polynomial variables must be unique canonical names")
        if any(
            len(term.exponents) != len(self.variables)
            for term in (*self.left, *self.right)
        ):
            raise ValueError("every exponent vector must match variable count")
        return self


class NormalizedModularPolynomialTerm(StrictModel):
    coefficient: StrictInt = Field(ge=1, lt=_MAX_MODULUS)
    exponents: tuple[StrictInt, ...] = Field(min_length=1, max_length=_MAX_VARIABLES)


class ModularPolynomialIdentityValue(StrictModel):
    modulus: StrictInt = Field(ge=2, le=_MAX_MODULUS)
    variable_order: tuple[StrictStr, ...] = Field(
        min_length=1, max_length=_MAX_VARIABLES
    )
    normalized_left: tuple[NormalizedModularPolynomialTerm, ...] = Field(
        max_length=_MAX_TERMS
    )
    normalized_right: tuple[NormalizedModularPolynomialTerm, ...] = Field(
        max_length=_MAX_TERMS
    )
    residual: tuple[NormalizedModularPolynomialTerm, ...] = Field(
        max_length=_MAX_TERMS * 2
    )
    identical: StrictBool
    comparison_scope: Literal["FORMAL_COEFFICIENTWISE_IDENTITY"] = (
        "FORMAL_COEFFICIENTWISE_IDENTITY"
    )

    @model_validator(mode="after")
    def bind_residual(self) -> Self:
        if self.identical != (not self.residual):
            raise ValueError("identity decision must match residual")
        for terms in (self.normalized_left, self.normalized_right, self.residual):
            exponents = [term.exponents for term in terms]
            if exponents != sorted(set(exponents)):
                raise ValueError("normalized terms must be unique and sorted")
            if any(
                len(term.exponents) != len(self.variable_order)
                or term.coefficient >= self.modulus
                for term in terms
            ):
                raise ValueError("normalized term is outside result scope")
        expected = _subtract_normalized(
            self.normalized_left, self.normalized_right, self.modulus
        )
        if self.residual != expected:
            raise ValueError("residual must equal normalized modular difference")
        return self


def _normalize(
    terms: tuple[ModularPolynomialTerm, ...], modulus: int
) -> tuple[NormalizedModularPolynomialTerm, ...]:
    coefficients: dict[tuple[int, ...], int] = {}
    for term in terms:
        coefficients[term.exponents] = (
            coefficients.get(term.exponents, 0) + int(term.coefficient)
        ) % modulus
    return tuple(
        NormalizedModularPolynomialTerm(coefficient=coefficient, exponents=exponents)
        for exponents, coefficient in sorted(coefficients.items())
        if coefficient
    )


def _subtract_normalized(
    left: tuple[NormalizedModularPolynomialTerm, ...],
    right: tuple[NormalizedModularPolynomialTerm, ...],
    modulus: int,
) -> tuple[NormalizedModularPolynomialTerm, ...]:
    coefficients = {term.exponents: term.coefficient for term in left}
    for term in right:
        coefficients[term.exponents] = (
            coefficients.get(term.exponents, 0) - term.coefficient
        ) % modulus
    return tuple(
        NormalizedModularPolynomialTerm(coefficient=coefficient, exponents=exponents)
        for exponents, coefficient in sorted(coefficients.items())
        if coefficient
    )


def modular_polynomial_identity(
    request: ModularPolynomialIdentityRequest,
) -> ModularPolynomialIdentityValue:
    left = _normalize(request.left, request.modulus)
    right = _normalize(request.right, request.modulus)
    residual = _subtract_normalized(left, right, request.modulus)
    return ModularPolynomialIdentityValue(
        modulus=request.modulus,
        variable_order=request.variables,
        normalized_left=left,
        normalized_right=right,
        residual=residual,
        identical=not residual,
    )


__all__ = [
    "ModularPolynomialIdentityRequest",
    "ModularPolynomialIdentityValue",
    "ModularPolynomialTerm",
    "NormalizedModularPolynomialTerm",
    "modular_polynomial_identity",
]
