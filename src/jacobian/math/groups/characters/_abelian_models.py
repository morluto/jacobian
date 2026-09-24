"""Typed exact character tables for bounded finite Abelian products."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.groups.characters._models import CyclotomicValue
from jacobian.math.groups.finite_abelian import FiniteAbelianProductGroup

MAX_ABELIAN_CHARACTER_TABLE_ORDER = 128
MAX_ABELIAN_CHARACTER_TABLE_CELLS = 16_384


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"groups.characters.abelian.{reason}", message)


class FiniteAbelianCharacterTableRequest(StrictModel):
    """Complete table request for one explicit product of cyclic groups."""

    group: FiniteAbelianProductGroup


class FiniteAbelianCharacterRow(StrictModel):
    """One linear character, identified by its dual product coordinates."""

    frequency: tuple[int, ...] = Field(min_length=0, max_length=6)
    values: tuple[CyclotomicValue, ...] = Field(
        min_length=1, max_length=MAX_ABELIAN_CHARACTER_TABLE_ORDER
    )


class FiniteAbelianCharacterTableResult(StrictModel):
    """Complete exact character table on the canonical product coordinates.

    Rows and columns both use lexicographic coordinate order.  The row indexed
    by ``frequency`` evaluates at ``x`` as zeta_E raised to
    ``sum(frequency[j] * x[j] * (E / modulus[j]))``, where E is the group
    exponent.
    """

    group: FiniteAbelianProductGroup
    elements: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_ABELIAN_CHARACTER_TABLE_ORDER
    )
    rows: tuple[FiniteAbelianCharacterRow, ...] = Field(
        min_length=1, max_length=MAX_ABELIAN_CHARACTER_TABLE_ORDER
    )
    cyclotomic_order: int = Field(ge=1, le=60)

    @model_validator(mode="after")
    def require_complete_shape(self) -> Self:
        count = self.group.order
        if len(self.elements) != count or len(self.rows) != count:
            raise _error(
                "table_shape",
                "table must contain every group element and dual character",
            )
        if any(len(element) != len(self.group.moduli) for element in self.elements):
            raise _error("element_rank", "each table element must match the group rank")
        if any(
            len(row.frequency) != len(self.group.moduli) or len(row.values) != count
            for row in self.rows
        ):
            raise _error(
                "row_shape",
                "each row must carry a dual coordinate and one value per element",
            )
        if any(
            value.order != self.cyclotomic_order
            for row in self.rows
            for value in row.values
        ):
            raise _error(
                "cyclotomic_axis",
                "all table values must use the declared cyclotomic order",
            )
        return self
