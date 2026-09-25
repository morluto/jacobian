"""Typed requests for character-valued degeneracy maps."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, ValidationError, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.matrices.cyclic_linear._models import RationalCyclotomicField
from jacobian.math.number_theory.characters.values import DirichletCharacter
from jacobian.math.number_theory.modular_forms.values import (
    ModularCharacterSpaceInclusion,
    ModularFormCoordinates,
    ModularFormFieldQExpansion,
    ModularFormSpace,
)


class ModularCharacterVDegeneracyRequest(StrictModel):
    """Apply V_d using an explicitly represented character-space inclusion."""

    form: ModularFormCoordinates
    target_space: ModularFormSpace


class ModularCharacterVDegeneracyImage(StrictModel):
    """An exact character-form image under ``V_d`` with its known q-prefix.

    The source and degeneracy index define the image as a modular form. The
    inclusion and finite prefix retain its exact ambient target and the
    coefficients determined by the represented source form.
    """

    source_form: ModularFormCoordinates
    d: StrictInt = Field(ge=2, le=3)
    inclusion: ModularCharacterSpaceInclusion
    q_expansion: ModularFormFieldQExpansion

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_form: ModularFormCoordinates,
        d: int,
        inclusion: ModularCharacterSpaceInclusion,
        q_expansion: ModularFormFieldQExpansion,
    ) -> Self:
        """Construct after source, map, and prefix admission in the owner."""

        return cls.model_construct(
            source_form=source_form,
            d=d,
            inclusion=inclusion,
            q_expansion=q_expansion,
        )

    @model_validator(mode="after")
    def require_exact_image_binding(self) -> Self:
        try:
            source_form = ModularFormCoordinates.model_validate(
                self.source_form.model_dump()
            )
            from jacobian.math.number_theory.modular_forms.space_maps import (
                require_modular_character_space_inclusion,
            )

            inclusion = require_modular_character_space_inclusion(self.inclusion)
            expansion = ModularFormFieldQExpansion.model_validate(
                self.q_expansion.model_dump()
            )
        except (AttributeError, TypeError, ValidationError, ValueError) as error:
            raise PydanticCustomError(
                "modular_forms.character_v_image_shape",
                "V_d image must contain canonical source, inclusion, and q-prefix values",
            ) from error
        if (
            source_form != self.source_form
            or inclusion != self.inclusion
            or expansion != self.q_expansion
        ):
            raise PydanticCustomError(
                "modular_forms.character_v_image_shape",
                "V_d image nested values must have canonical exact representations",
            )
        source = source_form.space
        target = inclusion.target_space
        character = source.character
        if (
            source.level != 13
            or source.weight != 2
            or source.kind != "S"
            or source.coefficient_domain != RationalCyclotomicField(order=6)
            or source_form.basis_id != "gamma0-13-even-order6-character-sturm-v1"
            or len(source_form.coordinates) != 1
            or type(character) is not DirichletCharacter
            or character.group.modulus != 13
            or character.coordinates not in ((2,), (10,))
            or inclusion.source_space != source
            or target.level != source.level * self.d
            or target.weight != source.weight
            or target.kind != source.kind
            or target.coefficient_domain != source.coefficient_domain
            or expansion.space != target
            or len(expansion.coefficients) != 2 * self.d + 1
        ):
            raise PydanticCustomError(
                "modular_forms.character_v_image_binding",
                "V_d image must retain its source, inflated target, and determined prefix",
            )
        return self


__all__ = [
    "ModularCharacterVDegeneracyImage",
    "ModularCharacterVDegeneracyRequest",
]
