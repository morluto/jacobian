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
    """Source-bound formal-polynomial identity claims.

    Deserialization checks only the canonical bounded representation.  The
    residual relation and identity decision are producer claims; consumers
    establish them with :func:`verify_modular_polynomial_identity`.
    """

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
    def require_structural_shape(self) -> Self:
        _validate_identity_value_shape(self)
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


def _validate_identity_value_shape(
    value: ModularPolynomialIdentityValue,
) -> None:
    """Validate the bounded canonical representation of an identity claim."""

    if not 2 <= value.modulus <= _MAX_MODULUS:
        raise ValueError("identity modulus is outside result scope")
    if not 1 <= len(value.variable_order) <= _MAX_VARIABLES:
        raise ValueError("identity variable axes are outside result scope")
    if len(set(value.variable_order)) != len(value.variable_order) or any(
        type(name) is not str or _VARIABLE.fullmatch(name) is None
        for name in value.variable_order
    ):
        raise ValueError("identity variable axes must be unique canonical names")
    if len(value.normalized_left) > _MAX_TERMS:
        raise ValueError("normalized_left exceeds the bounded term count")
    if len(value.normalized_right) > _MAX_TERMS:
        raise ValueError("normalized_right exceeds the bounded term count")
    if len(value.residual) > _MAX_TERMS * 2:
        raise ValueError("residual exceeds the bounded term count")

    for terms in (value.normalized_left, value.normalized_right, value.residual):
        if any(type(term) is not NormalizedModularPolynomialTerm for term in terms):
            raise ValueError("identity terms must use the canonical normalized type")
        exponents = [term.exponents for term in terms]
        if exponents != sorted(set(exponents)):
            raise ValueError("normalized terms must be unique and sorted")
        if any(
            len(term.exponents) != len(value.variable_order)
            or any(
                type(exponent) is not int
                or exponent < 0
                or exponent > _MAX_EXPONENT
                for exponent in term.exponents
            )
            or type(term.coefficient) is not int
            or not 1 <= term.coefficient < value.modulus
            for term in terms
        ):
            raise ValueError("normalized term is outside result scope")


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


def verify_modular_polynomial_identity(
    claim: ModularPolynomialIdentityValue,
) -> bool:
    """Check a supplied residual and identity decision against their sources.

    Structural admission bounds subtraction work to the two source term lists
    and its output to ``2 * _MAX_TERMS`` terms; no producer operation is
    replayed.
    """

    if not isinstance(claim, ModularPolynomialIdentityValue):
        return False
    try:
        _validate_identity_value_shape(claim)
        residual = _subtract_normalized(
            claim.normalized_left, claim.normalized_right, claim.modulus
        )
        return claim.residual == residual and claim.identical == (not residual)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


__all__ = [
    "ModularPolynomialIdentityValue",
    "ModularPolynomialTerm",
    "NormalizedModularPolynomialTerm",
    "modular_polynomial_identity",
    "verify_modular_polynomial_identity",
]
