"""Typed wire contracts for finite geometry operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_DIM = 32
MAX_FIELD_ORDER = 251


def _validate_prime_field(field_order: int) -> None:
    from sympy import isprime

    if not isprime(field_order):
        raise ValueError("field_order must be prime")


def _validate_nonzero_vector(vector: tuple[int, ...], field_order: int) -> None:
    if not vector:
        raise ValueError("vector must be nonempty")
    if len(vector) > MAX_DIM:
        raise ValueError("vector dimension exceeds bound")
    if any(not 0 <= v < field_order for v in vector):
        raise ValueError("vector entries must be canonical field residues")


class ProjectivePointCanonicalizeRequest(StrictModel):
    field_order: int = Field(ge=2, le=251)
    vector: tuple[int, ...] = Field(min_length=1, max_length=MAX_DIM)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_prime_field(self.field_order)
        _validate_nonzero_vector(self.vector, self.field_order)
        if all(v == 0 for v in self.vector):
            raise ValueError("projective point vector must be nonzero")
        return self


class ProjectivePointEqualRequest(StrictModel):
    field_order: int = Field(ge=2, le=251)
    vector_a: tuple[int, ...] = Field(min_length=1, max_length=MAX_DIM)
    vector_b: tuple[int, ...] = Field(min_length=1, max_length=MAX_DIM)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_prime_field(self.field_order)
        _validate_nonzero_vector(self.vector_a, self.field_order)
        _validate_nonzero_vector(self.vector_b, self.field_order)
        if len(self.vector_a) != len(self.vector_b):
            raise ValueError("vectors must have the same dimension")
        if all(v == 0 for v in self.vector_a):
            raise ValueError("vector_a must be nonzero")
        if all(v == 0 for v in self.vector_b):
            raise ValueError("vector_b must be nonzero")
        return self


class SubspaceComputeRequest(StrictModel):
    field_order: int = Field(ge=2, le=251)
    vectors: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_prime_field(self.field_order)
        width = len(self.vectors[0])
        if width == 0 or width > MAX_DIM:
            raise ValueError("vectors must have between 1 and 32 entries")
        if any(len(v) != width for v in self.vectors):
            raise ValueError("all vectors must have the same dimension")
        if any(not 0 <= entry < self.field_order for v in self.vectors for entry in v):
            raise ValueError("vector entries must be canonical field residues")
        return self


class SubspaceMembershipRequest(StrictModel):
    field_order: int = Field(ge=2, le=251)
    generators: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=16)
    word: tuple[int, ...] = Field(min_length=1, max_length=MAX_DIM)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_prime_field(self.field_order)
        width = len(self.generators[0])
        if width == 0 or width > MAX_DIM:
            raise ValueError("generators must have between 1 and 32 entries")
        if any(len(v) != width for v in self.generators):
            raise ValueError("all generators must have the same dimension")
        if len(self.word) != width:
            raise ValueError("word dimension must match ambient dimension")
        if any(
            not 0 <= entry < self.field_order for v in self.generators for entry in v
        ):
            raise ValueError("generator entries must be canonical field residues")
        if any(not 0 <= v < self.field_order for v in self.word):
            raise ValueError("word entries must be canonical field residues")
        return self


class SubspaceSpanRequest(StrictModel):
    field_order: int = Field(ge=2, le=251)
    vectors: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_prime_field(self.field_order)
        width = len(self.vectors[0])
        if width == 0 or width > MAX_DIM:
            raise ValueError("vectors must have between 1 and 32 entries")
        if any(len(v) != width for v in self.vectors):
            raise ValueError("all vectors must have the same dimension")
        if any(not 0 <= entry < self.field_order for v in self.vectors for entry in v):
            raise ValueError("vector entries must be canonical field residues")
        return self


class SubspaceIntersectionRequest(StrictModel):
    field_order: int = Field(ge=2, le=251)
    generators_a: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=16)
    generators_b: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_prime_field(self.field_order)
        width_a = len(self.generators_a[0])
        if width_a == 0 or width_a > MAX_DIM:
            raise ValueError("generators must have between 1 and 32 entries")
        width_b = len(self.generators_b[0])
        if width_b == 0 or width_b > MAX_DIM:
            raise ValueError("generators must have between 1 and 32 entries")
        if width_a != width_b:
            raise ValueError(
                "both subspace generator sets must have the same ambient dimension"
            )
        if any(len(v) != width_a for v in self.generators_a):
            raise ValueError("generators_a rows must have equal length")
        if any(len(v) != width_b for v in self.generators_b):
            raise ValueError("generators_b rows must have equal length")
        if any(
            not 0 <= entry < self.field_order for v in self.generators_a for entry in v
        ):
            raise ValueError("generators_a entries must be canonical field residues")
        if any(
            not 0 <= entry < self.field_order for v in self.generators_b for entry in v
        ):
            raise ValueError("generators_b entries must be canonical field residues")
        return self


class GrassmannianCountRequest(StrictModel):
    field_order: int = Field(ge=2, le=251)
    ambient_dimension: int = Field(ge=1, le=MAX_DIM)
    subspace_dimension: int = Field(ge=0, le=MAX_DIM)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_prime_field(self.field_order)
        if self.subspace_dimension > self.ambient_dimension:
            raise ValueError("subspace dimension cannot exceed ambient dimension")
        return self


class ProjectiveSpaceEnumerateRequest(StrictModel):
    field_order: int = Field(ge=2, le=251)
    projective_dimension: int = Field(ge=0, le=8)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_prime_field(self.field_order)
        total = self.field_order ** (self.projective_dimension + 1)
        if total > 65536:
            raise ValueError("projective space too large to enumerate")
        return self


# Results


class ProjectivePointCanonicalizeResult(StrictModel):
    canonical_vector: tuple[int, ...]
    scale: int = Field(ge=1)
    dimension: int = Field(ge=1)
    method: str = "FIRST_NONZERO_TO_ONE"


class ProjectivePointEqualResult(StrictModel):
    equal: bool
    scale: int = Field(ge=0)
    method: str = "SCALAR_MULTIPLE"


class SubspaceComputeResult(StrictModel):
    basis: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=0)
    ambient_dimension: int = Field(ge=1)
    method: str = "RREF"


class SubspaceMembershipResult(StrictModel):
    is_member: bool
    dimension: int = Field(ge=0)
    method: str = "RREF_MEMBERSHIP"


class SubspaceSpanResult(StrictModel):
    basis: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=0)
    ambient_dimension: int = Field(ge=1)
    method: str = "RREF"


class SubspaceIntersectionResult(StrictModel):
    basis: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=0)
    ambient_dimension: int = Field(ge=1)
    method: str = "INTERSECTION"


class GrassmannianCountResult(StrictModel):
    count: int = Field(ge=1)
    method: str = "GAUSSIAN_BINOMIAL"


class ProjectiveSpaceEnumerateResult(StrictModel):
    points: tuple[tuple[int, ...], ...]
    count: int = Field(ge=1)
    method: str = "CANONICAL_REPRESENTATIVES"
