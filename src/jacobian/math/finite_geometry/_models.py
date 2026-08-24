"""Typed wire contracts for finite geometry operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from sympy import isprime

from jacobian._models import StrictModel

MAX_DIM = 32
MAX_FIELD_ORDER = 10000


class PrimeFieldVectorSpace(StrictModel):
    """An ordered coordinate space over a named prime field."""

    field_order: int = Field(ge=2, le=MAX_FIELD_ORDER)
    axis: tuple[str, ...] = Field(min_length=1, max_length=MAX_DIM)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if not isprime(self.field_order):
            raise ValueError("field_order must be prime")
        if any(not label or not label.isidentifier() for label in self.axis):
            raise ValueError("axis labels must be nonempty identifiers")
        if len(set(self.axis)) != len(self.axis):
            raise ValueError("axis labels must be unique")
        return self


def _validate_vector(vector: tuple[int, ...], space: PrimeFieldVectorSpace) -> None:
    if len(vector) != len(space.axis):
        raise ValueError("vector length must match the ambient axis")
    if any(not 0 <= value < space.field_order for value in vector):
        raise ValueError("vector entries must be canonical field residues")


class ProjectivePoint(StrictModel):
    """A canonical point in a specific prime-field projective space."""

    space: PrimeFieldVectorSpace
    coordinates: tuple[int, ...]

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        _validate_vector(self.coordinates, self.space)
        try:
            first = next(value for value in self.coordinates if value != 0)
        except StopIteration as exc:
            raise ValueError("projective coordinates must be nonzero") from exc
        if first != 1:
            raise ValueError("projective coordinates must have first nonzero entry one")
        return self


class LinearSubspace(StrictModel):
    """A subspace represented by its unique RREF basis in an ordered parent."""

    space: PrimeFieldVectorSpace
    basis: tuple[tuple[int, ...], ...] = Field(max_length=32)

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        for row in self.basis:
            _validate_vector(row, self.space)
        pivots: list[int] = []
        for row_index, row in enumerate(self.basis):
            try:
                pivot = next(index for index, value in enumerate(row) if value != 0)
            except StopIteration as exc:
                raise ValueError("RREF basis cannot contain a zero row") from exc
            if row[pivot] != 1 or (pivots and pivot <= pivots[-1]):
                raise ValueError("basis must be in reduced row echelon form")
            if any(
                other[pivot] != 0
                for other_index, other in enumerate(self.basis)
                if other_index != row_index
            ):
                raise ValueError("basis must be in reduced row echelon form")
            pivots.append(pivot)
        return self

    @property
    def dimension(self) -> int:
        return len(self.basis)


class ProjectivePointCanonicalizeRequest(StrictModel):
    space: PrimeFieldVectorSpace
    vector: tuple[int, ...]

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_vector(self.vector, self.space)
        if all(value == 0 for value in self.vector):
            raise ValueError("projective point vector must be nonzero")
        return self


class ProjectivePointEqualRequest(StrictModel):
    point_a: ProjectivePoint
    point_b: ProjectivePoint

    @model_validator(mode="after")
    def require_same_parent(self) -> Self:
        if self.point_a.space != self.point_b.space:
            raise ValueError("projective points must have the same field and axis")
        return self


class SubspaceComputeRequest(StrictModel):
    space: PrimeFieldVectorSpace
    vectors: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        for vector in self.vectors:
            _validate_vector(vector, self.space)
        return self


class SubspaceMembershipRequest(StrictModel):
    subspace: LinearSubspace
    vector: tuple[int, ...]

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_vector(self.vector, self.subspace.space)
        return self


class SubspaceSpanRequest(StrictModel):
    space: PrimeFieldVectorSpace
    vectors: tuple[tuple[int, ...], ...] = Field(max_length=32)
    subspaces: tuple[LinearSubspace, ...] = Field(max_length=32)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if not self.vectors and not self.subspaces:
            raise ValueError("span requires at least one vector or subspace")
        for vector in self.vectors:
            _validate_vector(vector, self.space)
        if any(subspace.space != self.space for subspace in self.subspaces):
            raise ValueError("all subspaces must have the declared field and axis")
        if len(self.vectors) + sum(len(item.basis) for item in self.subspaces) > 32:
            raise ValueError("span generator count exceeds bound")
        return self


class SubspaceIntersectionRequest(StrictModel):
    subspace_a: LinearSubspace
    subspace_b: LinearSubspace

    @model_validator(mode="after")
    def require_same_parent(self) -> Self:
        if self.subspace_a.space != self.subspace_b.space:
            raise ValueError("subspaces must have the same field and axis")
        return self


class GrassmannianCountRequest(StrictModel):
    field_order: int = Field(ge=2, le=MAX_FIELD_ORDER)
    ambient_dimension: int = Field(ge=1, le=MAX_DIM)
    subspace_dimension: int = Field(ge=0, le=MAX_DIM)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if not isprime(self.field_order):
            raise ValueError("field_order must be prime")
        if self.subspace_dimension > self.ambient_dimension:
            raise ValueError("subspace dimension cannot exceed ambient dimension")
        return self


class ProjectiveSpaceEnumerateRequest(StrictModel):
    space: PrimeFieldVectorSpace

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if self.space.field_order ** len(self.space.axis) > 65536:
            raise ValueError("projective space too large to enumerate")
        return self


class ProjectivePointCanonicalizeResult(ProjectivePointCanonicalizeRequest):
    point: ProjectivePoint
    scale: int = Field(ge=1)
    method: str = "FIRST_NONZERO_TO_ONE"

    @model_validator(mode="after")
    def replay(self) -> Self:
        expected_scale = next(value for value in self.vector if value != 0)
        expected = tuple(
            value
            * pow(expected_scale, -1, self.space.field_order)
            % self.space.field_order
            for value in self.vector
        )
        if (
            self.point.space != self.space
            or self.point.coordinates != expected
            or self.scale != expected_scale
        ):
            raise ValueError("projective point is not the canonicalized source vector")
        return self


class ProjectivePointEqualResult(StrictModel):
    point_a: ProjectivePoint
    point_b: ProjectivePoint
    equal: bool
    method: str = "CANONICAL_REPRESENTATIVE"

    @model_validator(mode="after")
    def replay(self) -> Self:
        if self.point_a.space != self.point_b.space:
            raise ValueError("projective points must have the same field and axis")
        if self.equal != (self.point_a.coordinates == self.point_b.coordinates):
            raise ValueError("projective equality must match canonical coordinates")
        return self


class SubspaceComputeResult(SubspaceComputeRequest):
    subspace: LinearSubspace
    method: str = "RREF"

    @model_validator(mode="after")
    def replay(self) -> Self:
        from jacobian.math.finite_geometry._operations import _canonical_basis

        expected = tuple(
            tuple(row)
            for row in _canonical_basis(
                [list(vector) for vector in self.vectors], self.space.field_order
            )
        )
        if self.subspace.space != self.space or self.subspace.basis != expected:
            raise ValueError("subspace is not the span of its source vectors")
        return self


class SubspaceMembershipResult(StrictModel):
    subspace: LinearSubspace
    vector: tuple[int, ...]
    is_member: bool
    method: str = "RREF_MEMBERSHIP"

    @model_validator(mode="after")
    def replay(self) -> Self:
        from jacobian.math.finite_geometry._operations import _canonical_basis

        q = self.subspace.space.field_order
        enlarged = [list(row) for row in self.subspace.basis] + [list(self.vector)]
        expected = len(_canonical_basis(enlarged, q)) == self.subspace.dimension
        if self.is_member != expected:
            raise ValueError("membership does not match the bound subspace and vector")
        return self


class SubspaceSpanResult(SubspaceSpanRequest):
    subspace: LinearSubspace
    method: str = "RREF"

    @model_validator(mode="after")
    def replay(self) -> Self:
        from jacobian.math.finite_geometry._operations import _canonical_basis

        generators = [list(vector) for vector in self.vectors]
        generators.extend(
            list(row) for subspace in self.subspaces for row in subspace.basis
        )
        expected = tuple(
            tuple(row) for row in _canonical_basis(generators, self.space.field_order)
        )
        if self.subspace.space != self.space or self.subspace.basis != expected:
            raise ValueError("span result is not bound to its source values")
        return self


class SubspaceIntersectionResult(SubspaceIntersectionRequest):
    subspace: LinearSubspace
    method: str = "INTERSECTION"

    @model_validator(mode="after")
    def replay(self) -> Self:
        from jacobian.math.finite_geometry._operations import _intersection_basis

        expected = _intersection_basis(self.subspace_a, self.subspace_b)
        if (
            self.subspace.space != self.subspace_a.space
            or self.subspace.basis != expected
        ):
            raise ValueError("intersection is not bound to its source subspaces")
        return self


class GrassmannianCountResult(StrictModel):
    field_order: int
    ambient_dimension: int
    subspace_dimension: int
    count: int = Field(ge=1)
    method: str = "GAUSSIAN_BINOMIAL"

    @model_validator(mode="after")
    def replay(self) -> Self:
        if not isprime(self.field_order) or not (
            0 <= self.subspace_dimension <= self.ambient_dimension <= MAX_DIM
        ):
            raise ValueError("Grassmannian parameters are outside the public domain")
        numerator = 1
        denominator = 1
        for index in range(self.subspace_dimension):
            numerator *= self.field_order ** (self.ambient_dimension - index) - 1
            denominator *= self.field_order ** (self.subspace_dimension - index) - 1
        if self.count != numerator // denominator:
            raise ValueError("count does not match its Gaussian-binomial parameters")
        return self


class ProjectiveSpaceEnumerateResult(StrictModel):
    space: PrimeFieldVectorSpace
    points: tuple[ProjectivePoint, ...]
    count: int = Field(ge=1)
    method: str = "CANONICAL_REPRESENTATIVES"

    @model_validator(mode="after")
    def replay(self) -> Self:
        expected = (self.space.field_order ** len(self.space.axis) - 1) // (
            self.space.field_order - 1
        )
        if self.count != len(self.points) or self.count != expected:
            raise ValueError("projective point enumeration does not match its parent")
        if any(point.space != self.space for point in self.points):
            raise ValueError("every projective point must belong to the result space")
        if len({point.coordinates for point in self.points}) != len(self.points):
            raise ValueError("projective points must be unique")
        return self
