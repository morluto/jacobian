"""Typed diagonal-action tuple-family orbit profiles."""

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.groups._models import MAX_GROUP_DEGREE, PermutationGroup


class TupleFamilyOrbitSource(StrictModel):
    group: PermutationGroup
    arity: StrictInt = Field(ge=0, le=MAX_GROUP_DEGREE)
    family: tuple[tuple[StrictInt, ...], ...] = Field(max_length=4096)

    @model_validator(mode="after")
    def bind_family_axis(self) -> Self:
        if any(len(member) != self.arity for member in self.family):
            raise ValueError("every family member must have the declared arity")
        if any(
            not 0 <= coordinate < self.group.degree
            for member in self.family
            for coordinate in member
        ):
            raise ValueError("tuple coordinates must lie on the group axis")
        return self


class TupleOrbitRow(StrictModel):
    representative: tuple[StrictInt, ...]
    source_indices: tuple[StrictInt, ...]
    orbit_size: ExactInteger
    stabilizer_size: ExactInteger
    least_transporter: tuple[StrictInt, ...]


class TupleFamilyOrbitResult(StrictModel):
    source: TupleFamilyOrbitSource
    rows: tuple[TupleOrbitRow, ...] = Field(max_length=4096)
    is_union_of_complete_ambient_orbits: bool
