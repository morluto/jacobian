"""Typed wire contracts for exact bounded principal Dirichlet characters."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    Field,
    StrictBool,
    StrictInt,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.number_theory.characters.values import (
    MAX_PRINCIPAL_CHARACTER_MODULUS,
    PrincipalDirichletCharacter,
)

MAX_INTEGER_DIGITS = 256


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by Dirichlet-character contracts."""

    return PydanticCustomError(f"dirichlet_character.{reason}", message)


def _require_bounded_digits(value: str) -> str:
    if len(value.lstrip("-")) > MAX_INTEGER_DIGITS:
        raise _validation_error(
            "integer_digit_bound",
            f"integer exceeds the {MAX_INTEGER_DIGITS}-digit bound",
        )
    return value


BoundedCanonicalInteger = Annotated[
    CanonicalInteger,
    AfterValidator(_require_bounded_digits),
]


class PrincipalDirichletCharacterRequest(StrictModel):
    """Materialize the complete principal-character table for one modulus."""

    modulus: StrictInt = Field(
        ge=1,
        le=MAX_PRINCIPAL_CHARACTER_MODULUS,
        description=(
            "Positive modulus; the complete residue table has exactly this many "
            "entries and is bounded to 2,048 entries."
        ),
    )


class PrincipalDirichletCharacterValueRequest(StrictModel):
    """Evaluate a canonical principal-character value at one exact integer."""

    character: PrincipalDirichletCharacter
    integer: BoundedCanonicalInteger = Field(
        description="Canonical base-10 integer syntax, reduced modulo character.modulus."
    )

    def integer_value(self) -> int:
        return int(self.integer)


class PrincipalDirichletCharacterValueResult(StrictModel):
    """A source-bound principal-character evaluation with canonical residue data."""

    character: PrincipalDirichletCharacter
    integer: BoundedCanonicalInteger
    canonical_residue: StrictInt = Field(ge=0, lt=MAX_PRINCIPAL_CHARACTER_MODULUS)
    is_unit: StrictBool
    value: Literal[0, 1]

    @model_validator(mode="after")
    def require_exact_source_bound_value(self) -> Self:
        residue = int(self.integer) % self.character.modulus
        if self.canonical_residue != residue:
            raise _validation_error(
                "canonical_residue_mismatch",
                "canonical residue does not match the source integer",
            )
        expected_is_unit = math.gcd(residue, self.character.modulus) == 1
        if self.is_unit != expected_is_unit:
            raise _validation_error(
                "unit_status_mismatch",
                "unit status does not match the source character modulus",
            )
        expected_value = self.character.values[residue]
        if self.value != expected_value:
            raise _validation_error(
                "value_mismatch",
                "value does not match the source principal-character table",
            )
        return self


__all__ = [
    "MAX_INTEGER_DIGITS",
    "BoundedCanonicalInteger",
    "PrincipalDirichletCharacterRequest",
    "PrincipalDirichletCharacterValueRequest",
    "PrincipalDirichletCharacterValueResult",
]
