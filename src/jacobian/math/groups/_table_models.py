"""Bounded indexed multiplication-table values for exact finite groups."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_FINITE_TABLE_GROUP_ORDER = 24
"""Largest order admitted by the cubic group-law check (24^3 = 13,824)."""


FiniteGroupTableIndex = Annotated[
    StrictInt,
    Field(ge=0, le=MAX_FINITE_TABLE_GROUP_ORDER - 1),
]
FiniteGroupTableRow = Annotated[
    tuple[FiniteGroupTableIndex, ...],
    Field(min_length=1, max_length=MAX_FINITE_TABLE_GROUP_ORDER),
]


class FiniteGroupTableRequest(StrictModel):
    """An indexed finite multiplication table and its proposed identity.

    Elements are indexed by ``0..order-1``. This is a concrete table, not an
    isomorphism-canonical encoding: changing element indices changes the value.
    """

    multiplication: tuple[FiniteGroupTableRow, ...] = Field(
        min_length=1, max_length=MAX_FINITE_TABLE_GROUP_ORDER
    )
    identity: StrictInt = Field(ge=0, le=MAX_FINITE_TABLE_GROUP_ORDER - 1)


class FiniteGroupTable(StrictModel):
    """A bounded exact finite group represented by its complete indexed table."""

    multiplication: tuple[FiniteGroupTableRow, ...] = Field(
        min_length=1, max_length=MAX_FINITE_TABLE_GROUP_ORDER
    )
    identity: StrictInt = Field(ge=0, le=MAX_FINITE_TABLE_GROUP_ORDER - 1)
    inverse: tuple[FiniteGroupTableIndex, ...] = Field(
        min_length=1, max_length=MAX_FINITE_TABLE_GROUP_ORDER
    )

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:
        order = len(self.multiplication)
        if (
            any(len(row) != order for row in self.multiplication)
            or len(self.inverse) != order
            or self.identity >= order
            or any(value >= order for row in self.multiplication for value in row)
            or any(index >= order for index in self.inverse)
        ):
            raise PydanticCustomError(
                "finite_group.table.carrier_shape",
                "table, identity, and inverse indices must match one finite group order",
            )
        if any(
            self.multiplication[index][self.inverse[index]] != self.identity
            or self.multiplication[self.inverse[index]][index] != self.identity
            for index in range(order)
        ):
            raise PydanticCustomError(
                "finite_group.table.inverse_binding",
                "each inverse index must be a two-sided inverse in the multiplication table",
            )
        return self



class FiniteGroupTableElement(StrictModel):
    """One element index bound to its complete finite-group table."""

    group: FiniteGroupTable
    index: StrictInt = Field(ge=0, le=MAX_FINITE_TABLE_GROUP_ORDER - 1)

    @model_validator(mode="after")
    def require_element_in_parent(self) -> Self:
        if self.index >= len(self.group.multiplication):
            raise PydanticCustomError(
                "finite_group.table.element_parent",
                "element index must belong to its table group",
            )
        return self

class FiniteGroupTableResult(StrictModel):
    """A validated group table with its canonical parent-bound identity."""

    group: FiniteGroupTable
    identity_element: FiniteGroupTableElement

    @model_validator(mode="after")
    def require_identity_binding(self) -> Self:
        if (
            self.identity_element.group != self.group
            or self.identity_element.index != self.group.identity
        ):
            raise PydanticCustomError(
                "finite_group.table.result_binding",
                "identity element must be bound to the returned group",
            )
        return self
