"""Source-bound finite subsets and occupied affine-coset partitions."""

from typing import Annotated, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.geometry.finite._models import LinearSubspace
from jacobian.math.geometry.finite.values import PrimeFieldVectorSpace, _validate_vector

Vector = Annotated[tuple[int, ...], Field(max_length=32)]
Vectors = Annotated[tuple[Vector, ...], Field(max_length=65_536)]


class CosetIntersectionSource(StrictModel):
    """A lexicographically ordered finite subset and a supplied RREF subspace."""

    space: PrimeFieldVectorSpace
    subspace: LinearSubspace
    subset: Vectors

    @model_validator(mode="after")
    def require_structure(self) -> Self:
        if self.space != self.subspace.space:
            raise ValueError("subspace must have the declared field and ordered axis")
        for vector in self.subset:
            _validate_vector(vector, self.space)
        if any(a >= b for a, b in zip(self.subset, self.subset[1:], strict=False)):
            raise ValueError("subset must be strictly lexicographically ordered")
        return self


class CosetIntersection(StrictModel):
    """One occupied coset, with its complete source intersection."""

    representative: Vector
    members: Vectors = Field(min_length=1)
    cardinality: int = Field(ge=1, le=65_536)

    @model_validator(mode="after")
    def require_cardinality_matches_members(self) -> Self:
        if self.cardinality != len(self.members):
            raise ValueError("cardinality must equal the number of retained members")
        if any(a >= b for a, b in zip(self.members, self.members[1:], strict=False)):
            raise ValueError("members must be strictly lexicographically ordered")
        return self


class CosetIntersectionProfile(CosetIntersectionSource):
    """Complete partition with zero pivot coordinates in each representative.

    Parsing preserves source structure. The operation establishes the supplied
    field and RREF claims, the coset relation, and completeness of the partition.
    """

    rows: tuple[CosetIntersection, ...] = Field(max_length=65_536)

    @model_validator(mode="after")
    def require_row_coordinates(self) -> Self:
        for row in self.rows:
            _validate_vector(row.representative, self.space)
            for vector in row.members:
                _validate_vector(vector, self.space)
        collected = tuple(
            sorted(member for row in self.rows for member in row.members)
        )
        if collected != self.subset:
            raise ValueError("row members must be the complete retained subset")
        return self
