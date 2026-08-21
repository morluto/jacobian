"""Typed wire contracts for finite group operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel

MAX_GROUP_DEGREE = 64


class PermutationGroupRequest(StrictModel):
    """A finite permutation group given by generator permutations on {0,...,n-1}."""

    degree: int = Field(ge=1, le=MAX_GROUP_DEGREE)
    generators: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_GROUP_DEGREE
    )

    @model_validator(mode="after")
    def require_valid_generators(self) -> Self:
        for perm in self.generators:
            if len(perm) != self.degree:
                raise ValueError("each generator must have length equal to degree")
            if sorted(perm) != list(range(self.degree)):
                raise ValueError("each generator must be a permutation of 0..n-1")
        return self


class GroupOrderResult(StrictModel):
    """The exact order of a finite permutation group."""

    order: CanonicalInteger
    method: Literal["SYMPY_SCHREIER_SIMS"] = "SYMPY_SCHREIER_SIMS"


class GroupElementOrderRequest(StrictModel):
    """One generator (or group element) whose order is requested."""

    degree: int = Field(ge=1, le=MAX_GROUP_DEGREE)
    generator: tuple[int, ...] = Field(min_length=1, max_length=MAX_GROUP_DEGREE)

    @model_validator(mode="after")
    def require_valid_generator(self) -> Self:
        if len(self.generator) != self.degree:
            raise ValueError("generator must have length equal to degree")
        if sorted(self.generator) != list(range(self.degree)):
            raise ValueError("generator must be a permutation of 0..n-1")
        return self


class GroupElementOrderResult(StrictModel):
    """The exact order of one permutation."""

    order: CanonicalInteger


class GroupOrbitRequest(StrictModel):
    """Request the orbit of a point under a permutation group."""

    degree: int = Field(ge=1, le=MAX_GROUP_DEGREE)
    generators: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_GROUP_DEGREE
    )
    point: int = Field(ge=0, le=MAX_GROUP_DEGREE - 1)

    @model_validator(mode="after")
    def require_valid_generators(self) -> Self:
        if not 0 <= self.point < self.degree:
            raise ValueError("point must be in 0..degree-1")
        for perm in self.generators:
            if len(perm) != self.degree:
                raise ValueError("each generator must have length equal to degree")
            if sorted(perm) != list(range(self.degree)):
                raise ValueError("each generator must be a permutation of 0..n-1")
        return self


class GroupOrbitResult(StrictModel):
    """The orbit of a point under a permutation group."""

    orbit: tuple[int, ...] = Field(min_length=1, max_length=MAX_GROUP_DEGREE)
    point: int = Field(ge=0, le=MAX_GROUP_DEGREE - 1)


class GroupStabilizerRequest(StrictModel):
    """Request generators of the stabilizer of a point in a permutation group."""

    degree: int = Field(ge=1, le=MAX_GROUP_DEGREE)
    generators: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_GROUP_DEGREE
    )
    point: int = Field(ge=0, le=MAX_GROUP_DEGREE - 1)

    @model_validator(mode="after")
    def require_valid_generators(self) -> Self:
        if not 0 <= self.point < self.degree:
            raise ValueError("point must be in 0..degree-1")
        for perm in self.generators:
            if len(perm) != self.degree:
                raise ValueError("each generator must have length equal to degree")
            if sorted(perm) != list(range(self.degree)):
                raise ValueError("each generator must be a permutation of 0..n-1")
        return self


def _require_permutation(perm: tuple[int, ...], degree: int, label: str) -> None:
    if len(perm) != degree:
        raise ValueError(f"each {label} must have length equal to degree")
    if sorted(perm) != list(range(degree)):
        raise ValueError(f"each {label} must be a permutation of 0..n-1")


def _check_stabilizer_permutations(
    degree: int,
    point: int,
    generators: tuple[tuple[int, ...], ...],
    source_generators: tuple[tuple[int, ...], ...],
) -> None:
    if not 0 <= point < degree:
        raise ValueError("point must be in 0..degree-1")
    for perm in source_generators:
        _require_permutation(perm, degree, "source generator")
    for perm in generators:
        _require_permutation(perm, degree, "generator")
        if perm[point] != point:
            raise ValueError("stabilizer generators must fix the point")


def _check_orbit_stabilizer(
    degree: int,
    point: int,
    generators: tuple[tuple[int, ...], ...],
    source_generators: tuple[tuple[int, ...], ...],
) -> None:
    from sympy.combinatorics import Permutation, PermutationGroup

    source_perms = [Permutation(list(p)) for p in source_generators]
    source_group = PermutationGroup(source_perms)
    source_order = int(source_group.order())
    orbit = source_group.orbit(point)
    orbit_size = len(orbit) if orbit is not None else 1
    if generators:
        stab_perms = [Permutation(list(p)) for p in generators]
        stab_group = PermutationGroup(stab_perms)
        stab_order = int(stab_group.order())
        for perm in generators:
            if not source_group.contains(Permutation(list(perm))):
                raise ValueError(
                    "stabilizer generators must be elements of the source group"
                )
        if source_order % stab_order != 0:
            raise ValueError("stabilizer order must divide source order")
    else:
        stab_order = 1
    if source_order != orbit_size * stab_order:
        raise ValueError(
            "orbit size times stabilizer order must equal source order "
            "(orbit-stabilizer theorem)"
        )


class GroupStabilizerResult(StrictModel):
    """Generators of the point stabilizer subgroup, bound to its source.

    The result retains the ambient ``degree`` and the source group's
    generators so the stabilizer is interpretable as a canonical
    permutation-group value (``degree`` + ``generators``) without
    reattaching external context. Validation replays the defining
    orbit-stabilizer relation ``|G| = |orbit(point)| * |Stab(point)|``
    and checks that every stabilizer generator fixes ``point`` and lies
    in the source group.
    """

    degree: int = Field(ge=1, le=MAX_GROUP_DEGREE)
    point: int = Field(ge=0, le=MAX_GROUP_DEGREE - 1)
    generators: tuple[tuple[int, ...], ...] = Field(
        default=(), max_length=MAX_GROUP_DEGREE
    )
    source_generators: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_GROUP_DEGREE
    )
    method: Literal["SYMPY_STABILIZER"] = "SYMPY_STABILIZER"

    @model_validator(mode="after")
    def require_valid_stabilizer(self) -> Self:
        _check_stabilizer_permutations(
            self.degree, self.point, self.generators, self.source_generators
        )
        try:
            _check_orbit_stabilizer(
                self.degree, self.point, self.generators, self.source_generators
            )
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover - unexpected backend failure
            raise ValueError(f"stabilizer validation failed: {exc}") from exc
        return self
