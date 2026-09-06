"""Formal coefficientwise polynomial arithmetic modulo one integer."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    StringConstraints,
    WithJsonSchema,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue

from jacobian._exact import DecimalIntegerEncoding
from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError

_MAX_INTEGER_DIGITS = 256
_MAX_MODULUS = 1_000_000
_MAX_VARIABLES = 20
_MAX_TERMS = 512
_MAX_EXPONENT = 256
_VARIABLE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
ModularPolynomialCoefficient = Annotated[
    int, DecimalIntegerEncoding(max_digits=_MAX_INTEGER_DIGITS)
]


class ModularPolynomialTerm(StrictModel):
    coefficient: ModularPolynomialCoefficient
    exponents: tuple[StrictInt, ...] = Field(min_length=1, max_length=_MAX_VARIABLES)

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


IdentityVariableName = Annotated[
    StrictStr,
    StringConstraints(
        pattern=r"^[a-z][a-z0-9_]{0,31}$",
        max_length=32,
        strict=True,
    ),
]


def _identity_normalized_term_schema() -> JsonSchemaValue:
    """Project the shared term carrier onto the identity result envelope."""

    schema = NormalizedModularPolynomialTerm.model_json_schema()
    exponents = schema["properties"]["exponents"]
    assert isinstance(exponents, dict)
    items = exponents["items"]
    assert isinstance(items, dict)
    items["minimum"] = 0
    items["maximum"] = _MAX_EXPONENT
    return schema


IdentityNormalizedModularPolynomialTerm = Annotated[
    NormalizedModularPolynomialTerm,
    WithJsonSchema(_identity_normalized_term_schema()),
]


class ModularPolynomialIdentityValue(StrictModel):
    """Source-bound formal-polynomial identity claims.

    Deserialization checks only the canonical bounded representation.  The
    residual relation and identity decision are producer claims; consumers
    establish them with :func:`verify_modular_polynomial_identity`.
    """

    modulus: StrictInt = Field(ge=2, le=_MAX_MODULUS)
    variable_order: tuple[IdentityVariableName, ...] = Field(
        min_length=1,
        max_length=_MAX_VARIABLES,
        json_schema_extra={"uniqueItems": True},
    )
    normalized_left: tuple[IdentityNormalizedModularPolynomialTerm, ...] = Field(
        max_length=_MAX_TERMS
    )
    normalized_right: tuple[IdentityNormalizedModularPolynomialTerm, ...] = Field(
        max_length=_MAX_TERMS
    )
    residual: tuple[IdentityNormalizedModularPolynomialTerm, ...] = Field(
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


def _validate_identity_axes(value: ModularPolynomialIdentityValue) -> None:
    """Validate exact axes before hashing their names."""

    if type(value.variable_order) is not tuple:
        raise ValueError("identity variable axes must use a plain tuple")
    if not 1 <= len(value.variable_order) <= _MAX_VARIABLES:
        raise ValueError("identity variable axes are outside result scope")
    if any(type(name) is not str for name in value.variable_order):
        raise ValueError("identity variable axes must use plain strings")
    if any(_VARIABLE.fullmatch(name) is None for name in value.variable_order):
        raise ValueError("identity variable axes must be canonical names")
    if len(set(value.variable_order)) != len(value.variable_order):
        raise ValueError("identity variable axes must be unique canonical names")


def _validate_identity_terms(
    terms: object,
    *,
    axis_count: int,
    modulus: int,
    maximum: int,
    label: str,
) -> None:
    """Validate one term sequence before sorting exponent tuples."""

    if type(terms) is not tuple or len(terms) > maximum:
        raise ValueError(f"{label} exceeds the bounded term count")
    if any(type(term) is not NormalizedModularPolynomialTerm for term in terms):
        raise ValueError("identity terms must use the canonical normalized type")
    if any(type(term.exponents) is not tuple for term in terms):
        raise ValueError("normalized exponents must use plain tuples")
    if any(len(term.exponents) != axis_count for term in terms):
        raise ValueError("normalized exponent dimensions must match the axes")
    if any(
        type(exponent) is not int or exponent < 0 or exponent > _MAX_EXPONENT
        for term in terms
        for exponent in term.exponents
    ):
        raise ValueError("normalized exponents are outside result scope")
    if any(
        type(term.coefficient) is not int or not 1 <= term.coefficient < modulus
        for term in terms
    ):
        raise ValueError("normalized coefficients are outside result scope")
    exponents = [term.exponents for term in terms]
    if exponents != sorted(set(exponents)):
        raise ValueError("normalized terms must be unique and sorted")


def _validate_identity_value_shape(
    value: ModularPolynomialIdentityValue,
) -> None:
    """Validate the bounded canonical representation of an identity claim."""

    if type(value.modulus) is not int or not 2 <= value.modulus <= _MAX_MODULUS:
        raise ValueError("identity modulus is outside result scope")
    _validate_identity_axes(value)
    if (
        type(value.normalized_left) is not tuple
        or type(value.normalized_right) is not tuple
        or type(value.residual) is not tuple
    ):
        raise ValueError("identity term sequences must use plain tuples")
    if type(value.identical) is not bool:
        raise ValueError("identity decision must be a boolean claim")
    if type(value.comparison_scope) is not str or value.comparison_scope != (
        "FORMAL_COEFFICIENTWISE_IDENTITY"
    ):
        raise ValueError("identity comparison scope is not canonical")
    axis_count = len(value.variable_order)
    _validate_identity_terms(
        value.normalized_left,
        axis_count=axis_count,
        modulus=value.modulus,
        maximum=_MAX_TERMS,
        label="normalized_left",
    )
    _validate_identity_terms(
        value.normalized_right,
        axis_count=axis_count,
        modulus=value.modulus,
        maximum=_MAX_TERMS,
        label="normalized_right",
    )
    _validate_identity_terms(
        value.residual,
        axis_count=axis_count,
        modulus=value.modulus,
        maximum=_MAX_TERMS * 2,
        label="residual",
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
