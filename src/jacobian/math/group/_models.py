"""Typed wire contracts for finite group operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel

MAX_GROUP_DEGREE = 64

# Conjugacy classes materialize every group element as an array form
# (``|G|`` permutations of length ``degree``).  The degree cap alone does
# not bound this enumeration: S12 already has 479M elements.  Use a
# conservative group-order bound derived before backend expansion via
# Schreier-Sims (cheap) rather than after enumeration.
MAX_CONJUGACY_CLASSES_GROUP_ORDER = 5000


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


class GroupConjugacyClassesRequest(StrictModel):
    """Request the conjugacy classes of a permutation group."""

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
        # Bound enumeration by the generated group's order before invoking
        # the SymPy backend which materializes every element.  The degree
        # cap alone does not bound work: S12 has order 479M and would
        # exhaust memory; compute |G| cheaply via Schreier-Sims and reject
        # when it exceeds the conservative limit.
        from sympy.combinatorics import Permutation, PermutationGroup

        perms = [Permutation(list(p)) for p in self.generators]
        group = PermutationGroup(perms)
        order = int(group.order())
        if order > MAX_CONJUGACY_CLASSES_GROUP_ORDER:
            raise ValueError(
                f"group order {order} exceeds the bounded maximum "
                f"{MAX_CONJUGACY_CLASSES_GROUP_ORDER} for conjugacy classes "
                f"(would materialize |G|={order} elements)"
            )
        return self


class GroupConjugacyClassesResult(StrictModel):
    """The conjugacy classes of a permutation group as array forms."""

    classes: tuple[tuple[tuple[int, ...], ...], ...]
    method: Literal["SYMPY_CONJUGACY_CLASSES"] = "SYMPY_CONJUGACY_CLASSES"

    @model_validator(mode="after")
    def require_nonempty_classes(self) -> Self:
        if not self.classes:
            raise ValueError("conjugacy classes must be nonempty")
        for cls in self.classes:
            if not cls:
                raise ValueError("each conjugacy class must be nonempty")
        return self
