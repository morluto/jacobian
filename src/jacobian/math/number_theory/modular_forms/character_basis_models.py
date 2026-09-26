"""Typed exact carriers for the bounded character-valued basis slice."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters.values import DirichletCharacter
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)


class ModularCharacterBasisRequest(StrictModel):
    """Request one supported exact character-valued basis prefix."""

    space: ModularFormSpace
    precision: StrictInt | None = Field(default=None, ge=1, le=128)


class ModularCharacterQExpansion(StrictModel):
    """One finite q-prefix over the explicit cyclotomic field of its space."""

    space: ModularFormSpace
    basis_id: Literal[
        "gamma0-13-even-order6-character-sturm-v1",
        "gamma0-cyclotomic-character-sturm-rref-v1",
    ]
    coefficients: tuple[RationalCyclotomicElement, ...] = Field(
        min_length=1, max_length=128
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
    """Apply one admitted Hecke index to a character-valued form."""

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
    precision: StrictInt = Field(ge=1, le=128)
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


class CyclotomicCharacterMap(StrictModel):
    """Explicit pullback of one Dirichlet character along level reduction."""

    source: DirichletCharacter
    target: DirichletCharacter


class CyclotomicIdentityFieldMap(StrictModel):
    """The explicit identity embedding between identical cyclotomic parents."""

    source: RationalCyclotomicField
    target: RationalCyclotomicField


class ModularCharacterSpaceInclusion(StrictModel):
    """A represented nested-level inclusion with its character and field maps."""

    source_space: ModularFormSpace
    target_space: ModularFormSpace
    character_map: CyclotomicCharacterMap
    coefficient_field_map: CyclotomicIdentityFieldMap

    @model_validator(mode="after")
    def require_structural_compatibility(self) -> Self:
        source = self.source_space
        target = self.target_space
        if (
            source.character != self.character_map.source
            or target.character != self.character_map.target
            or source.coefficient_domain != self.coefficient_field_map.source
            or target.coefficient_domain != self.coefficient_field_map.target
            or source.weight != target.weight
            or source.kind != "S"
            or target.kind != "S"
            or source.level <= 0
            or target.level % source.level
            or source.coefficient_domain != target.coefficient_domain
            or self.character_map.source.group.modulus != source.level
            or self.character_map.target.group.modulus != target.level
        ):
            raise PydanticCustomError(
                "modular_forms.character_inclusion_binding",
                "character inclusion parents, map values, weight, kind, levels, and fields must agree",
            )
        return self


class ModularCharacterCommonTargetPrefix(StrictModel):
    """An exact q-prefix in an explicitly embedded character space."""

    space: ModularFormSpace
    precision: StrictInt = Field(ge=1, le=128)
    coefficients: tuple[RationalCyclotomicElement, ...] = Field(
        min_length=1, max_length=128
    )

    @model_validator(mode="after")
    def require_exact_parent_and_precision(self) -> Self:
        field = self.space.coefficient_domain
        if (
            type(field) is not RationalCyclotomicField
            or len(self.coefficients) != self.precision
            or any(value.field != field for value in self.coefficients)
        ):
            raise PydanticCustomError(
                "modular_forms.character_common_prefix_parent",
                "common-target prefix must have exact precision and the declared cyclotomic parent",
            )
        return self


class ModularCharacterTransportedForm(StrictModel):
    """A source form, its explicit inclusion, and exact target representation."""

    source_form: ModularFormCoordinates
    inclusion: ModularCharacterSpaceInclusion
    target_form: ModularFormCoordinates | None
    target_q_expansion: ModularCharacterQExpansion | ModularCharacterCommonTargetPrefix

    @model_validator(mode="after")
    def require_transport_binding(self) -> Self:
        if (
            self.source_form.space != self.inclusion.source_space
            or self.target_q_expansion.space != self.inclusion.target_space
            or len(self.target_q_expansion.coefficients) == 0
            or (
                isinstance(self.target_q_expansion, ModularCharacterQExpansion)
                and (
                    self.target_form is None
                    or self.target_form.space != self.inclusion.target_space
                    or self.target_q_expansion.basis_id != self.target_form.basis_id
                )
            )
            or (
                isinstance(self.target_q_expansion, ModularCharacterCommonTargetPrefix)
                and self.target_form is not None
            )
        ):
            raise PydanticCustomError(
                "modular_forms.character_transport_binding",
                "transported form must retain and bind exact source and target parents",
            )
        return self


class ModularCharacterCoordinatesTransportRequest(StrictModel):
    form: ModularFormCoordinates
    inclusion: ModularCharacterSpaceInclusion


class ModularCharacterEqualityRequest(StrictModel):
    left: ModularCharacterTransportedForm
    right: ModularCharacterTransportedForm


class ModularCharacterEqualityResult(StrictModel):
    equal: StrictBool


__all__ = [
    "CyclotomicCharacterMap",
    "CyclotomicIdentityFieldMap",
    "ModularCharacterBasis",
    "ModularCharacterBasisElement",
    "ModularCharacterBasisRequest",
    "ModularCharacterCommonTargetPrefix",
    "ModularCharacterCoordinatesRequest",
    "ModularCharacterCoordinatesTransportRequest",
    "ModularCharacterEqualityRequest",
    "ModularCharacterEqualityResult",
    "ModularCharacterHeckeMatrix",
    "ModularCharacterHeckeMatrixRequest",
    "ModularCharacterHeckeRequest",
    "ModularCharacterQExpansion",
    "ModularCharacterSpaceInclusion",
    "ModularCharacterTransportedForm",
]
