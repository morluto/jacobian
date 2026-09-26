"""Typed request and result for multidimensional character Hecke matrices."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.modular_forms.values import ModularFormSpace


class ModularCharacterHeckeMatrixRequest(StrictModel):
    """Compute T_n on a multidimensional exact character space."""

    space: ModularFormSpace
    index: StrictInt = Field(ge=1, le=8)


class ModularCharacterHeckeMatrixResult(StrictModel):
    """T_n in the canonical q-Sturm RREF basis.

    ``entries[row][column]`` is the coefficient of the row-labeled basis
    vector in T_n applied to the column-labeled basis vector.
    """

    space: ModularFormSpace
    basis_id: Literal["gamma0-cyclotomic-character-sturm-rref-v1"]
    index: StrictInt = Field(ge=1, le=8)
    labels: tuple[str, ...] = Field(min_length=2, max_length=32)
    entries: tuple[tuple[RationalCyclotomicElement, ...], ...] = Field(
        min_length=2, max_length=32
    )

    @model_validator(mode="after")
    def require_square_matrix_in_source_field(self) -> Self:
        field = self.space.coefficient_domain
        if (
            type(field) is not RationalCyclotomicField
            or len(set(self.labels)) != len(self.labels)
            or len(self.entries) != len(self.labels)
            or any(len(row) != len(self.labels) for row in self.entries)
            or any(value.field != field for row in self.entries for value in row)
        ):
            raise PydanticCustomError(
                "modular_forms.multidimensional_hecke_matrix_parent",
                "the Hecke matrix must be square, labeled, and over its exact space field",
            )
        return self
