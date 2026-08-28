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
from jacobian.catalog.models import OperationDomainValidationError

_MAX_INTEGER_DIGITS = 256
_MAX_MODULUS = 1_000_000
_MAX_VARIABLES = 20
_MAX_TERMS = 512
_MAX_EXPONENT = 256
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
            raise ValueError("term exponents must be between 0 and 256")
        return self


class ModularPolynomialIdentityRequest(StrictModel):
    modulus: StrictInt = Field(ge=2, le=_MAX_MODULUS)
    variables: tuple[StrictStr, ...] = Field(min_length=1, max_length=_MAX_VARIABLES)
    left: tuple[ModularPolynomialTerm, ...] = Field(default=(), max_length=_MAX_TERMS)
    right: tuple[ModularPolynomialTerm, ...] = Field(default=(), max_length=_MAX_TERMS)


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
        if self.residual != _subtract_normalized(
            self.normalized_left, self.normalized_right, self.modulus
        ):
            raise ValueError(
                "residual must equal normalized_left - normalized_right modulo modulus"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        modulus: int,
        variable_order: tuple[str, ...],
        normalized_left: tuple[NormalizedModularPolynomialTerm, ...],
        normalized_right: tuple[NormalizedModularPolynomialTerm, ...],
        residual: tuple[NormalizedModularPolynomialTerm, ...],
    ) -> Self:
        return cls.model_construct(
            modulus=modulus,
            variable_order=variable_order,
            normalized_left=normalized_left,
            normalized_right=normalized_right,
            residual=residual,
            identical=not residual,
            comparison_scope="FORMAL_COEFFICIENTWISE_IDENTITY",
        )


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


def _require_identity_admission(
    modulus: int,
    variables: tuple[str, ...],
    left: tuple[ModularPolynomialTerm, ...],
    right: tuple[ModularPolynomialTerm, ...],
) -> None:
    """Validate the owner-local formal-polynomial identity envelope."""

    if not 2 <= modulus <= _MAX_MODULUS:
        raise ValueError(f"modulus must be between 2 and {_MAX_MODULUS}")
    if not 1 <= len(variables) <= _MAX_VARIABLES:
        raise ValueError(f"variable count must be between 1 and {_MAX_VARIABLES}")
    if len(left) > _MAX_TERMS or len(right) > _MAX_TERMS:
        raise ValueError(f"each polynomial may contain at most {_MAX_TERMS} terms")
    if len(set(variables)) != len(variables) or any(
        _VARIABLE.fullmatch(name) is None for name in variables
    ):
        raise ValueError("polynomial variables must be unique canonical names")
    if any(len(term.exponents) != len(variables) for term in (*left, *right)):
        raise ValueError("every exponent vector must match variable count")


def _compute_modular_polynomial_identity(
    modulus: int,
    variables: tuple[str, ...],
    left: tuple[ModularPolynomialTerm, ...],
    right: tuple[ModularPolynomialTerm, ...],
) -> ModularPolynomialIdentityValue:
    """Compare already admitted canonical sparse polynomials."""

    normalized_left = _normalize(left, modulus)
    normalized_right = _normalize(right, modulus)
    residual = _subtract_normalized(normalized_left, normalized_right, modulus)
    return ModularPolynomialIdentityValue._from_kernel(
        modulus=modulus,
        variable_order=variables,
        normalized_left=normalized_left,
        normalized_right=normalized_right,
        residual=residual,
    )


def modular_polynomial_identity(
    modulus: int,
    variables: tuple[str, ...],
    left: tuple[ModularPolynomialTerm, ...] = (),
    right: tuple[ModularPolynomialTerm, ...] = (),
) -> ModularPolynomialIdentityValue:
    """Compare canonical sparse polynomials coefficientwise modulo ``modulus``.

    This native boundary accepts domain values and semantic scalars.  Catalog
    and MCP calls parse ``ModularPolynomialIdentityRequest`` once and use the
    private adapter below.
    """

    if type(modulus) is not int:
        raise TypeError("modulus must be an integer")
    if not isinstance(variables, tuple) or not all(
        type(variable) is str for variable in variables
    ):
        raise TypeError("variables must be a tuple of strings")
    for name, terms in (("left", left), ("right", right)):
        if not isinstance(terms, tuple) or not all(
            isinstance(term, ModularPolynomialTerm) for term in terms
        ):
            raise TypeError(f"{name} must be a tuple of ModularPolynomialTerm values")
    try:
        _require_identity_admission(modulus, variables, left, right)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("variables",),
            code="modular_polynomial.identity_domain",
            message=str(exc),
        ) from exc
    return _compute_modular_polynomial_identity(modulus, variables, left, right)


def _modular_polynomial_identity_request(
    request: ModularPolynomialIdentityRequest,
) -> ModularPolynomialIdentityValue:
    """Catalog adapter for the strict modular-polynomial identity request."""

    return modular_polynomial_identity(
        request.modulus, request.variables, request.left, request.right
    )


__all__ = [
    "ModularPolynomialIdentityValue",
    "ModularPolynomialTerm",
    "NormalizedModularPolynomialTerm",
    "modular_polynomial_identity",
]
