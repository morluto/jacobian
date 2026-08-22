"""Typed wire contracts for finite group operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

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
    """Request the conjugacy classes of a permutation group.

    The generated group must have order at most 5000 (degree up to 64
    alone does not bound enumeration; e.g., S8 has order 40320 and S12
    has order 479M). Validators compute ``|G|`` via Schreier-Sims and
    reject groups exceeding the 5000-element bound before enumeration.
    """

    degree: int = Field(
        ge=1,
        le=MAX_GROUP_DEGREE,
        description=(
            "Degree n of the permutation group acting on {0,...,n-1}; "
            "the generated group must have order at most 5000 (degree 64 alone "
            "does not bound enumeration)."
        ),
    )
    generators: tuple[tuple[int, ...], ...] = Field(
        min_length=1,
        max_length=MAX_GROUP_DEGREE,
        description=(
            "Generator permutations as array forms of length degree; "
            "the generated group must have order at most 5000."
        ),
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


ConjugacyClassElement = Annotated[
    tuple[int, ...],
    Field(min_length=1, max_length=MAX_GROUP_DEGREE),
]

ConjugacyClass = Annotated[
    tuple[ConjugacyClassElement, ...],
    Field(min_length=1, max_length=MAX_CONJUGACY_CLASSES_GROUP_ORDER),
]


class GroupConjugacyClassesResult(StrictModel):
    """The exact conjugacy-class partition of a permutation group.

    ``classes`` is canonically ordered (members lexicographically sorted,
    classes sorted by smallest member) so equal groups serialize
    identically. Validation replays the defining invariant: the listed
    elements must form a group under composition and each class must be
    exactly its representative's conjugation orbit in that group.
    """

    classes: tuple[ConjugacyClass, ...] = Field(
        min_length=1,
        max_length=MAX_CONJUGACY_CLASSES_GROUP_ORDER,
    )
    method: Literal["SYMPY_CONJUGACY_CLASSES"] = "SYMPY_CONJUGACY_CLASSES"

    @model_validator(mode="after")
    def require_conjugacy_class_partition(self) -> Self:
        elements = _require_canonical_partition(self.classes)

        from sympy.combinatorics import Permutation, PermutationGroup

        ordered_elements = sorted(elements)
        backend_group = PermutationGroup(
            [Permutation(list(p)) for p in ordered_elements]
        )
        # A finite subset of a symmetric group is a subgroup iff it is closed
        # under composition; <S> equals S iff |<S>| == |S|.
        if int(backend_group.order()) != len(ordered_elements):
            raise ValueError("the listed elements must form a group under composition")

        generators = _minimal_generator_subset(ordered_elements)
        inverse_generators = [_invert_permutation(p) for p in generators]
        for cls in self.classes:
            orbit = _conjugation_orbit(cls[0], generators, inverse_generators)
            if orbit != set(cls):
                raise ValueError(
                    "each class must equal the conjugation orbit of its "
                    "representative in the reconstructed group"
                )
        return self


def _require_canonical_partition(
    classes: tuple[ConjugacyClass, ...],
) -> set[tuple[int, ...]]:
    degree = len(classes[0][0])
    expected_range = list(range(degree))
    elements: set[tuple[int, ...]] = set()
    previous_representative: tuple[int, ...] | None = None
    for cls in classes:
        previous_member: tuple[int, ...] | None = None
        for perm in cls:
            if len(perm) != degree:
                raise ValueError("all permutations must have one common degree")
            if sorted(perm) != expected_range:
                raise ValueError("every element must be a permutation of 0..degree-1")
            if perm in elements:
                raise ValueError("conjugacy classes must not repeat an element")
            elements.add(perm)
            if previous_member is not None and perm <= previous_member:
                raise ValueError("class members must be strictly increasing")
            previous_member = perm
        representative = cls[0]
        if (
            previous_representative is not None
            and representative <= previous_representative
        ):
            raise ValueError("classes must be ordered by canonical representative")
        previous_representative = representative
    if len(elements) > MAX_CONJUGACY_CLASSES_GROUP_ORDER:
        raise ValueError(
            f"a conjugacy-class partition may contain at most "
            f"{MAX_CONJUGACY_CLASSES_GROUP_ORDER} elements"
        )
    return elements


def _minimal_generator_subset(
    elements: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    from sympy.combinatorics import Permutation, PermutationGroup

    total = len(elements)
    generators: list[tuple[int, ...]] = []
    generated_order = 1
    for element in elements:
        if generated_order == total:
            break
        candidate = [*generators, element]
        candidate_order = int(
            PermutationGroup([Permutation(list(p)) for p in candidate]).order()
        )
        if candidate_order > generated_order:
            generators = candidate
            generated_order = candidate_order
    return generators


def _conjugation_orbit(
    representative: tuple[int, ...],
    generators: list[tuple[int, ...]],
    inverse_generators: list[tuple[int, ...]],
) -> set[tuple[int, ...]]:
    orbit = {representative}
    stack = [representative]
    while stack:
        current = stack.pop()
        for generator, inverse in zip(generators, inverse_generators, strict=True):
            conjugate = _compose_permutation(
                _compose_permutation(generator, current), inverse
            )
            if conjugate not in orbit:
                orbit.add(conjugate)
                stack.append(conjugate)
    return orbit


def _compose_permutation(
    first: tuple[int, ...], second: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(first[i] for i in second)


def _invert_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for index, value in enumerate(permutation):
        inverse[value] = index
    return tuple(inverse)
