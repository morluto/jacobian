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


# The conjugacy result serializes every group element and the lattice
# traverses the subgroup structure; both stay bounded only under an
# explicit enumerated-order cap enforced at this typed boundary.
MAX_GROUP_ORDER = 64


def _require_bounded_group_order(
    degree: int, generators: tuple[tuple[int, ...], ...]
) -> None:
    from sympy.combinatorics import Permutation, PermutationGroup

    group = PermutationGroup(*(Permutation(list(g)) for g in generators))
    order = int(group.order())
    if order > MAX_GROUP_ORDER:
        raise ValueError(
            f"group order {order} exceeds the bounded maximum "
            f"{MAX_GROUP_ORDER} for full-element enumeration"
        )


class GroupConjugacyClassesRequest(StrictModel):
    """Compute the conjugacy classes of a permutation group."""

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
        _require_bounded_group_order(self.degree, self.generators)
        return self


class ConjugacyClass(StrictModel):
    """One conjugacy class with representative elements and size."""

    elements: tuple[tuple[int, ...], ...] = Field(min_length=1)
    size: int = Field(ge=1)

    @model_validator(mode="after")
    def require_size_matches_elements(self) -> Self:
        if self.size != len(self.elements) or len(set(self.elements)) != len(
            self.elements
        ):
            raise ValueError(
                "class size must equal the number of distinct retained elements"
            )
        return self


class GroupConjugacyClassesResult(StrictModel):
    """All conjugacy classes of a permutation group."""

    classes: tuple[ConjugacyClass, ...] = Field(min_length=1)
    class_count: int = Field(ge=1)
    method: Literal["SYMPY_CONJUGACY_CLASSES"] = "SYMPY_CONJUGACY_CLASSES"

    @model_validator(mode="after")
    def require_consistent_count(self) -> Self:
        if len(self.classes) != self.class_count:
            raise ValueError("class_count must match the number of classes")
        return self


class GroupSubgroupLatticeRequest(StrictModel):
    """Enumerate all subgroups of a bounded permutation group."""

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
        _require_bounded_group_order(self.degree, self.generators)
        return self


class SubgroupEntry(StrictModel):
    """One subgroup with its generators and order."""

    generators: tuple[tuple[int, ...], ...] = Field(min_length=1)
    order: int = Field(ge=1, le=64)

    @model_validator(mode="after")
    def require_order_matches_generators(self) -> Self:
        degree = len(self.generators[0])
        if any(
            len(gen) != degree or sorted(gen) != list(range(degree))
            for gen in self.generators
        ):
            raise ValueError(
                "each subgroup generator must be a permutation of equal degree"
            )
        if degree > MAX_GROUP_DEGREE:
            raise ValueError("subgroup generators exceed the supported degree bound")
        from sympy.combinatorics import Permutation, PermutationGroup

        group_order = int(
            PermutationGroup(*(Permutation(list(g)) for g in self.generators)).order()
        )
        if group_order != self.order:
            raise ValueError(
                f"subgroup order {self.order} does not match the order "
                f"{group_order} generated by the retained generators"
            )
        return self


class GroupSubgroupLatticeResult(StrictModel):
    """All subgroups of a bounded permutation group."""

    subgroups: tuple[SubgroupEntry, ...] = Field(min_length=1)
    subgroup_count: int = Field(ge=1)
    method: Literal["SYMPY_SUBGROUPS"] = "SYMPY_SUBGROUPS"

    @model_validator(mode="after")
    def require_consistent_count(self) -> Self:
        if len(self.subgroups) != self.subgroup_count:
            raise ValueError("subgroup_count must match the number of subgroups")
        return self
