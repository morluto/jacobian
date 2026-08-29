"""Typed contracts for binary-union relation hypergraph operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

MAX_FAMILY_SIZE = 200
MAX_SET_SIZE = 100
MAX_GROUND_SET_SIZE = 200


class BinaryUnionRelationRequest(StrictModel):
    """Request to compute the binary-union relation of a set family.

    The family is a tuple of sets, each represented as a sorted tuple of
    non-negative integers.
    """

    family: tuple[tuple[int, ...], ...] = Field(
        min_length=1,
        max_length=MAX_FAMILY_SIZE,
    )

    @model_validator(mode="after")
    def validate_family(self) -> Self:
        seen_sets: set[frozenset[int]] = set()
        for i, s in enumerate(self.family):
            if len(s) > MAX_SET_SIZE:
                raise PydanticCustomError(
                    "set_system.set_too_large",
                    f"set {i} exceeds the {MAX_SET_SIZE}-element limit",
                )
            if len(set(s)) != len(s):
                raise PydanticCustomError(
                    "set_system.duplicate_elements",
                    f"set {i} contains duplicate elements",
                )
            if list(s) != sorted(s):
                raise PydanticCustomError(
                    "set_system.elements_not_sorted",
                    f"set {i} elements must be sorted",
                )
            seen_sets.add(frozenset(s))
        if len(seen_sets) != len(self.family):
            raise PydanticCustomError(
                "set_system.duplicate_members",
                "family members must be pairwise distinct",
            )
        flat: set[int] = set()
        for s in self.family:
            flat.update(s)
        if len(flat) > MAX_GROUND_SET_SIZE:
            raise PydanticCustomError(
                "set_system.ground_set_too_large",
                f"ground set exceeds the {MAX_GROUND_SET_SIZE}-element limit",
            )
        return self


class UnionRelationRow(StrictModel):
    """One row of the binary-union relation."""

    operand_i: int
    operand_j: int
    result_k: int


class BinaryUnionRelationResult(StrictModel):
    """The complete binary-union relation of a set family."""

    family: tuple[tuple[int, ...], ...]
    rows: tuple[UnionRelationRow, ...]
    hypergraph: FiniteHypergraph


__all__ = [
    "MAX_FAMILY_SIZE",
    "MAX_GROUND_SET_SIZE",
    "MAX_SET_SIZE",
    "BinaryUnionRelationRequest",
    "BinaryUnionRelationResult",
    "UnionRelationRow",
]
