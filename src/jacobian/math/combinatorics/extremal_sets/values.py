"""Canonical values for finite indexed set families."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_VERTICES,
)

MAX_GROUND_SET_SIZE = (1 << 53) - 1
MAX_FAMILY_SIZE = MAX_VERTICES


def _value_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"set_system.{reason}", message)


class IndexedFiniteSetFamily(StrictModel):
    """A distinct indexed family of subsets of the declared axis ``[n]``.

    Member order is source identity. Each member is a strictly increasing tuple
    of indices in ``0,...,ground_set_size-1``. The family may be empty.
    """

    ground_set_size: StrictInt = Field(ge=0, le=MAX_GROUND_SET_SIZE)
    members: tuple[tuple[StrictInt, ...], ...] = Field(max_length=MAX_FAMILY_SIZE)

    @model_validator(mode="after")
    def require_canonical_family(self) -> Self:
        seen: set[tuple[int, ...]] = set()
        for member in self.members:
            if any(not 0 <= element < self.ground_set_size for element in member):
                raise _value_error(
                    "element_out_of_range",
                    "member elements must lie in 0..ground_set_size-1",
                )
            if tuple(sorted(member)) != member:
                raise _value_error(
                    "elements_not_sorted",
                    "each member must be a strictly increasing index tuple",
                )
            if len(set(member)) != len(member):
                raise _value_error(
                    "duplicate_elements", "a member must not repeat an element"
                )
            if member in seen:
                raise _value_error(
                    "duplicate_members", "family members must be pairwise distinct"
                )
            seen.add(member)
        return self


__all__ = ["MAX_FAMILY_SIZE", "MAX_GROUND_SET_SIZE", "IndexedFiniteSetFamily"]
