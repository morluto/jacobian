"""Typed exact carriers for the bounded character-valued basis slice."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)

MAX_CHARACTER_BASIS_PRECISION = 128


class ModularCharacterBasisRequest(StrictModel):
    """Request one supported exact character-valued basis prefix."""

    space: ModularFormSpace
    precision: StrictInt | None = Field(
        default=None,
        ge=1,
        le=MAX_CHARACTER_BASIS_PRECISION,
        description=(
            "Number of coefficients a_0 through a_(P-1); omission requests the "
            "smallest Sturm-determining prefix."
        ),
    )


class ModularCharacterQExpansion(StrictModel):
    """One finite q-prefix over the explicit cyclotomic field of its space."""

    space: ModularFormSpace
    basis_id: Literal[
        "gamma0-13-even-order6-character-sturm-v1",
        "gamma0-cyclotomic-character-sturm-rref-v1",
    ]
    coefficients: tuple[RationalCyclotomicElement, ...] = Field(
        min_length=1, max_length=MAX_CHARACTER_BASIS_PRECISION
    )

    @model_validator(mode="after")
    def require_coefficient_parent(self) -> Self:
        field = self.space.coefficient_domain
        if type(field) is not RationalCyclotomicField:
            raise PydanticCustomError(
                "modular_forms.character_q_parent",
                "character q-expansion requires a cyclotomic coefficient field",
            )
        if any(value.field != field for value in self.coefficients):
            raise PydanticCustomError(
                "modular_forms.character_q_coefficient_parent",
                "every q coefficient must belong to the space coefficient field",
            )
        return self


class ModularCharacterCoordinatesRequest(StrictModel):
    """Construct the exact Sturm prefix of one represented character form."""

    form: ModularFormCoordinates


class ModularCharacterHeckeRequest(StrictModel):
    """Apply one Hecke index, retaining the exact source space as target."""

    form: ModularFormCoordinates
    index: StrictInt = Field(ge=1, le=32)


class ModularCharacterHeckeMatrixRequest(StrictModel):
    """Request the Hecke action matrix in the canonical character basis."""

    space: ModularFormSpace
    index: StrictInt = Field(ge=1, le=32)


class ModularCharacterHeckeMatrix(StrictModel):
    """Exact Hecke matrix bound to one represented character space and basis."""

    space: ModularFormSpace
    basis_id: Literal["gamma0-13-even-order6-character-sturm-v1"]
    index: StrictInt = Field(ge=1, le=32)
    entries: tuple[tuple[RationalCyclotomicElement],] = Field(
        min_length=1, max_length=1
    )

    @model_validator(mode="after")
    def require_entry_parent(self) -> Self:
        field = self.space.coefficient_domain
        if (
            type(field) is not RationalCyclotomicField
            or self.entries[0][0].field != field
        ):
            raise PydanticCustomError(
                "modular_forms.character_hecke_matrix_parent",
                "the Hecke matrix entry must belong to its character space field",
            )
        return self


class ModularCharacterCoordinatesProductRequest(StrictModel):
    """Multiply the two conjugate represented character forms."""

    left: ModularFormCoordinates
    right: ModularFormCoordinates


class ModularCharacterBasisElement(StrictModel):
    label: str = Field(min_length=1, max_length=96)
    expansion: ModularCharacterQExpansion


class ModularCharacterBasis(StrictModel):
    """A complete admitted character-valued basis through a Sturm prefix."""

    space: ModularFormSpace
    basis_id: Literal[
        "gamma0-13-even-order6-character-sturm-v1",
        "gamma0-cyclotomic-character-sturm-rref-v1",
    ]
    precision: StrictInt = Field(ge=1, le=MAX_CHARACTER_BASIS_PRECISION)
    elements: tuple[ModularCharacterBasisElement, ...] = Field(max_length=32)

    @model_validator(mode="after")
    def require_source_and_precision(self) -> Self:
        if any(
            item.expansion.space != self.space
            or item.expansion.basis_id != self.basis_id
            or len(item.expansion.coefficients) != self.precision
            for item in self.elements
        ):
            raise PydanticCustomError(
                "modular_forms.character_basis_binding",
                "character basis vectors must preserve the exact source and precision",
            )
        if len(self.elements) > 32:
            raise PydanticCustomError(
                "modular_forms.character_basis_dimension",
                "character basis dimension exceeds the admitted coordinate bound",
            )
        return self


__all__ = [
    "MAX_CHARACTER_BASIS_PRECISION",
    "ModularCharacterBasis",
    "ModularCharacterBasisElement",
    "ModularCharacterBasisRequest",
    "ModularCharacterCoordinatesRequest",
    "ModularCharacterHeckeMatrix",
    "ModularCharacterHeckeMatrixRequest",
    "ModularCharacterHeckeRequest",
    "ModularCharacterQExpansion",
]
