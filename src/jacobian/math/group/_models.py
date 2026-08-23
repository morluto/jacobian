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


# Conjugacy classes serialize each element of the generated group exactly
# once; the subgroup-lattice traversal is exponential in that order and
# therefore carries a much tighter enumerated-order cap. Both stay bounded
# only under explicit caps enforced at this typed boundary.
MAX_CONJUGACY_CLASSES_GROUP_ORDER = 5000
MAX_SUBGROUP_LATTICE_GROUP_ORDER = 64


def _require_bounded_group_order(
    degree: int,
    generators: tuple[tuple[int, ...], ...],
    maximum: int,
    purpose: str,
) -> None:
    from sympy.combinatorics import Permutation, PermutationGroup

    group = PermutationGroup(*(Permutation(list(g)) for g in generators))
    order = int(group.order())
    if order > maximum:
        raise ValueError(
            f"group order {order} exceeds the bounded maximum {maximum} for {purpose}"
        )


class GroupConjugacyClassesRequest(StrictModel):
    """Compute the conjugacy classes of a permutation group.

    The generated group must have order at most 5000 (degree up to 64
    alone does not bound enumeration; e.g., S8 has order 40320). The
    order is computed via Schreier-Sims before any element enumeration.
    """

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
        _require_bounded_group_order(
            self.degree,
            self.generators,
            MAX_CONJUGACY_CLASSES_GROUP_ORDER,
            "conjugacy classes",
        )
        return self


class ConjugacyClass(StrictModel):
    """One conjugacy class with representative elements and size."""

    elements: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_CONJUGACY_CLASSES_GROUP_ORDER
    )
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
    """All conjugacy classes of a permutation group, bound to its source."""

    degree: int = Field(ge=1, le=MAX_GROUP_DEGREE)
    generators: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_GROUP_DEGREE
    )
    classes: tuple[ConjugacyClass, ...] = Field(
        min_length=1, max_length=MAX_CONJUGACY_CLASSES_GROUP_ORDER
    )
    class_count: int = Field(ge=1)
    method: Literal["SYMPY_CONJUGACY_CLASSES"] = "SYMPY_CONJUGACY_CLASSES"

    @model_validator(mode="after")
    def require_consistent_count(self) -> Self:
        if len(self.classes) != self.class_count:
            raise ValueError("class_count must match the number of classes")
        return self

    @model_validator(mode="after")
    def bind_classes_to_source_group(self) -> Self:
        # Replaying through the request model revalidates generator shape and
        # the bounded group order before the partition is recomputed.
        GroupConjugacyClassesRequest(degree=self.degree, generators=self.generators)
        from jacobian.math.group.operations import conjugacy_classes

        expected = tuple(
            (tuple(sorted(tuple(element) for element in class_elements)), size)
            for class_elements, size in conjugacy_classes(
                self.degree, [list(generator) for generator in self.generators]
            )
        )
        actual = tuple(
            (bound_class.elements, bound_class.size) for bound_class in self.classes
        )
        if actual != expected or self.class_count != len(expected):
            raise ValueError(
                "classes must be the exact conjugacy partition of the "
                "retained source group in canonical element and class order"
            )
        return self


class GroupSubgroupLatticeRequest(StrictModel):
    """Enumerate all subgroups of a bounded permutation group.

    The subgroup traversal is exponential in the generated group's order,
    so the lattice is bounded to groups of order at most 64.
    """

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
        _require_bounded_group_order(
            self.degree,
            self.generators,
            MAX_SUBGROUP_LATTICE_GROUP_ORDER,
            "subgroup lattice enumeration",
        )
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
    """All subgroups of a bounded permutation group, bound to its source."""

    degree: int = Field(ge=1, le=MAX_GROUP_DEGREE)
    generators: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_GROUP_DEGREE
    )
    subgroups: tuple[SubgroupEntry, ...] = Field(min_length=1)
    subgroup_count: int = Field(ge=1)
    method: Literal["SYMPY_SUBGROUPS"] = "SYMPY_SUBGROUPS"

    @model_validator(mode="after")
    def require_consistent_count(self) -> Self:
        if len(self.subgroups) != self.subgroup_count:
            raise ValueError("subgroup_count must match the number of subgroups")
        return self

    @model_validator(mode="after")
    def bind_lattice_to_source_group(self) -> Self:
        # Replaying through the request model revalidates generator shape and
        # the bounded group order before the lattice is recomputed.
        GroupSubgroupLatticeRequest(degree=self.degree, generators=self.generators)
        from jacobian.math.group.operations import subgroup_lattice

        expected_lattice = tuple(
            (tuple(tuple(g) for g in sorted(subgroup_generators)), order)
            for subgroup_generators, order in subgroup_lattice(
                self.degree, [list(generator) for generator in self.generators]
            )
        )
        actual_lattice = tuple(
            (entry.generators, entry.order) for entry in self.subgroups
        )
        if len(set(expected_lattice)) != len(expected_lattice):
            raise ValueError("subgroups must be distinct")
        if actual_lattice != expected_lattice or self.subgroup_count != len(
            expected_lattice
        ):
            raise ValueError(
                "subgroups must be the exact complete subgroup lattice of "
                "the retained source group in canonical element and entry order"
            )
        return self
