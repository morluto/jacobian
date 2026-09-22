"""Typed wire contracts for exact bounded principal Dirichlet characters."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    Field,
    StrictBool,
    StrictInt,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.number_theory.characters.values import (
    MAX_CHARACTER_GROUP_MODULUS,
    MAX_PRINCIPAL_CHARACTER_MODULUS,
    CyclotomicValue,
    DirichletCharacter,
    DirichletCharacterGroup,
    PrincipalDirichletCharacter,
)

MAX_INTEGER_DIGITS = 256


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by Dirichlet-character contracts."""

    return PydanticCustomError(f"dirichlet_character.{reason}", message)


def _require_bounded_digits(value: int) -> int:
    if len(format_canonical_integer(abs(value))) > MAX_INTEGER_DIGITS:
        raise _validation_error(
            "integer_digit_bound",
            f"integer exceeds the {MAX_INTEGER_DIGITS}-digit bound",
        )
    return value


DirichletCharacterInteger = Annotated[
    ExactInteger,
    AfterValidator(_require_bounded_digits),
]


class DirichletCharacterRequest(StrictModel):
    group: DirichletCharacterGroup
    coordinates: tuple[StrictInt, ...]


class DirichletCharacterValueRequest(StrictModel):
    character: DirichletCharacter
    integer: DirichletCharacterInteger


class DirichletCharacterValueResult(StrictModel):
    character: DirichletCharacter
    integer: DirichletCharacterInteger
    canonical_residue: StrictInt
    is_unit: StrictBool
    value: CyclotomicValue | None


class DirichletCharacterProductRequest(StrictModel):
    left: DirichletCharacter
    right: DirichletCharacter


class DirichletCharacterTableResult(StrictModel):
    character: DirichletCharacter
    residues: tuple[StrictInt, ...]
    values: tuple[CyclotomicValue | None, ...]


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


class CharacterGroupRequest(StrictModel):
    """Compute the finite unit-group decomposition for one modulus."""

    modulus: StrictInt = Field(
        ge=1,
        le=MAX_CHARACTER_GROUP_MODULUS,
        description=(
            "Positive modulus; the unit table and generator-coordinate maps "
            "cover exactly phi(modulus) unit residues within the 2,048-entry "
            "bound."
        ),
    )


class PrincipalDirichletCharacterValueRequest(StrictModel):
    """Evaluate a canonical principal-character value at one exact integer."""

    character: PrincipalDirichletCharacter
    integer: DirichletCharacterInteger = Field(
        description="Canonical base-10 integer syntax, reduced modulo character.modulus."
    )

    def integer_value(self) -> int:
        return int(self.integer)


class PrincipalDirichletCharacterValueResult(StrictModel):
    """A source-bound principal-character evaluation with canonical residue data."""

    character: PrincipalDirichletCharacter
    integer: DirichletCharacterInteger
    canonical_residue: StrictInt = Field(ge=0, lt=MAX_PRINCIPAL_CHARACTER_MODULUS)
    is_unit: StrictBool
    value: Literal[0, 1]

    @model_validator(mode="after")
    def require_source_bound_residue(self) -> Self:
        """Validate only the inexpensive source-to-residue structural binding."""

        residue = int(self.integer) % self.character.modulus
        if self.canonical_residue != residue:
            raise _validation_error(
                "canonical_residue_mismatch",
                "canonical residue does not match the source integer",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        character: PrincipalDirichletCharacter,
        integer: DirichletCharacterInteger,
        canonical_residue: StrictInt,
        is_unit: StrictBool,
        value: Literal[0, 1],
    ) -> Self:
        """Build a result after its producer has established the evaluation."""

        return cls.model_construct(
            character=character,
            integer=integer,
            canonical_residue=canonical_residue,
            is_unit=is_unit,
            value=value,
        )


__all__ = [
    "MAX_INTEGER_DIGITS",
    "CharacterGroupRequest",
    "DirichletCharacterInteger",
    "DirichletCharacterProductRequest",
    "DirichletCharacterRequest",
    "DirichletCharacterTableResult",
    "DirichletCharacterValueRequest",
    "DirichletCharacterValueResult",
    "PrincipalDirichletCharacterRequest",
    "PrincipalDirichletCharacterValueRequest",
    "PrincipalDirichletCharacterValueResult",
]
