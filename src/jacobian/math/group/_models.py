"""Typed wire contracts for finite group operations."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

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
# The subgroup-lattice traversal is exponential in the generated group's
# order and therefore carries a much tighter enumerated-order cap.
MAX_SUBGROUP_LATTICE_GROUP_ORDER = 64
# The complete lattice of an admitted source has at most as many entries as
# the extremal admitted group: the elementary abelian group C2^6, whose
# subgroups are exactly the subspaces of F_2^6 (a Gaussian-binomial sum
# 1 + 63 + 651 + 1395 + 651 + 63 + 1). Capping the relayed payload at this
# operation-derived count keeps result parsing bounded before any nested
# entry constructs a backend permutation group.
MAX_SUBGROUP_LATTICE_ENTRIES = 2825


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
    """Request the orbit of a point under a permutation group.

    The group is the canonical permutation-group value, so a stabilizer
    result's ``stabilizer`` subgroup feeds the advertised orbit consumer
    unchanged instead of being unpacked into parallel top-level fields.
    """

    group: PermutationGroupRequest = Field(
        description=(
            "Permutation group acting on {0,...,degree-1} as the canonical "
            "value; pass a previous stabilizer or order request's group "
            "value unchanged."
        )
    )
    point: int = Field(ge=0, le=MAX_GROUP_DEGREE - 1)

    @model_validator(mode="after")
    def require_point_in_group(self) -> Self:
        if not 0 <= self.point < self.group.degree:
            raise ValueError("point must be in 0..group.degree-1")
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


class GroupStabilizerRequest(StrictModel):
    """Request the stabilizer of a point in a permutation group.

    The group is the canonical permutation-group value, so a stabilizer chain
    passes a previous result's ``stabilizer`` subgroup unchanged as ``group``.
    """

    group: PermutationGroupRequest = Field(
        description=(
            "Permutation group acting on {0,...,degree-1} as the canonical "
            "value; pass a previous stabilizer result's `stabilizer` subgroup "
            "unchanged to continue a stabilizer chain."
        )
    )
    point: int = Field(ge=0, le=MAX_GROUP_DEGREE - 1)

    @model_validator(mode="after")
    def require_point_in_group(self) -> Self:
        if not 0 <= self.point < self.group.degree:
            raise ValueError("point must be in 0..group.degree-1")
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
    """Point stabilizer as a canonical permutation-group value bound to its source.

    The result retains the source group and the stabilizer subgroup as nested
    canonical :class:`PermutationGroupRequest` values (``degree`` + ``generators``)
    so the stabilizer can be passed unchanged to ``group.order.compute`` or any
    other permutation-group consumer without reshaping. The trivial stabilizer
    is represented by the identity permutation ``[0,...,degree-1]`` (the
    consumer requires at least one generator). Validation replays the defining
    orbit-stabilizer relation ``|G| = |orbit(point)| * |Stab(point)|`` and
    checks that every stabilizer generator fixes ``point`` and lies in the
    source group.
    """

    point: int = Field(
        ge=0,
        le=MAX_GROUP_DEGREE - 1,
        description="Point whose stabilizer is computed; must satisfy 0 <= point < source.degree == stabilizer.degree.",
    )
    source: PermutationGroupRequest = Field(
        description="Source permutation group as the canonical value (degree + generators)."
    )
    stabilizer: PermutationGroupRequest = Field(
        description=(
            "Stabilizer subgroup as a canonical permutation-group value on the same "
            "degree; trivial stabilizer is represented by the identity permutation "
            "[0,...,degree-1] so the value is always consumable by group.order.compute "
            "without reshaping or synthesizing an identity."
        )
    )
    method: Literal["SYMPY_STABILIZER"] = "SYMPY_STABILIZER"

    @model_validator(mode="after")
    def require_valid_stabilizer(self) -> Self:
        if self.source.degree != self.stabilizer.degree:
            raise ValueError("source and stabilizer must have the same degree")
        if not 0 <= self.point < self.source.degree:
            raise ValueError("point must be in 0..degree-1")
        _check_stabilizer_permutations(
            self.source.degree,
            self.point,
            self.stabilizer.generators,
            self.source.generators,
        )
        try:
            _check_orbit_stabilizer(
                self.source.degree,
                self.point,
                self.stabilizer.generators,
                self.source.generators,
            )
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover - unexpected backend failure
            raise ValueError(f"stabilizer validation failed: {exc}") from exc
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
    """One subgroup as the canonical permutation-group value plus its order.

    Carrying ``group: PermutationGroupRequest`` lets callers chain an
    enumerated subgroup into ``group.order.compute``, ``group.orbit.compute``,
    or any other permutation-group consumer unchanged.
    """

    group: PermutationGroupRequest
    order: int = Field(ge=1, le=MAX_SUBGROUP_LATTICE_GROUP_ORDER)

    @model_validator(mode="after")
    def require_order_matches_generators(self) -> Self:
        from sympy.combinatorics import Permutation, PermutationGroup

        group_order = int(
            PermutationGroup(
                *(Permutation(list(g)) for g in self.group.generators)
            ).order()
        )
        if group_order != self.order:
            raise ValueError(
                f"subgroup order {self.order} does not match the order "
                f"{group_order} generated by the retained generators"
            )
        return self


GroupSubgroupLatticeOutcome = Literal["COMPUTED", "LIMIT_EXCEEDED"]


class GroupSubgroupLatticeResult(StrictModel):
    """All subgroups of a bounded permutation group, bound to its source."""

    outcome: GroupSubgroupLatticeOutcome = "COMPUTED"
    degree: int = Field(ge=1, le=MAX_GROUP_DEGREE)
    generators: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_GROUP_DEGREE
    )
    subgroups: tuple[SubgroupEntry, ...] | None = None
    subgroup_count: int = Field(default=0, ge=0)
    method: Literal["SYMPY_SUBGROUPS"] = "SYMPY_SUBGROUPS"
    detail: str | None = None

    @model_validator(mode="before")
    @classmethod
    def require_bounded_subgroup_entries(cls, data: Any) -> Any:
        # Cap the entry count BEFORE nested SubgroupEntry construction: each
        # nested entry builds a backend permutation group, so a forged relayed
        # payload must not drive that work past the operation-derived bound.
        if isinstance(data, dict):
            entries = data.get("subgroups")
            if isinstance(entries, (list, tuple)) and len(entries) > (
                MAX_SUBGROUP_LATTICE_ENTRIES
            ):
                raise ValueError(
                    "subgroups exceed the admitted lattice bound of "
                    f"{MAX_SUBGROUP_LATTICE_ENTRIES} entries (the extremal "
                    "subgroup count among groups of order at most 64)"
                )
        return data

    @model_validator(mode="after")
    def require_outcome_shape(self) -> Self:
        if self.outcome == "COMPUTED":
            if self.subgroups is None or self.detail is not None:
                raise ValueError(
                    "computed lattice requires subgroup entries and no detail"
                )
            if self.subgroup_count != len(self.subgroups) or self.subgroup_count < 1:
                raise ValueError("subgroup_count must match the number of subgroups")
        elif self.subgroups is not None or self.detail is None:
            raise ValueError("an exceeded traversal carries only a safe detail")
        elif self.subgroup_count != 0:
            # No entries are retained on exhaustion, so any positive count
            # would fabricate an exact conclusion.
            raise ValueError("an exceeded traversal reports no subgroup count")
        return self

    @model_validator(mode="after")
    def bind_lattice_to_source_group(self) -> Self:
        # Replaying through the request model revalidates generator shape and
        # the bounded group order for EVERY outcome, including relayed
        # limit-exceeded payloads whose source must still be admissible.
        GroupSubgroupLatticeRequest(degree=self.degree, generators=self.generators)
        if self.outcome != "COMPUTED" or self.subgroups is None:
            return self
        from jacobian.math.group.operations import subgroup_lattice

        expected_lattice = tuple(
            (entry.group.generators, entry.order)
            for entry in subgroup_lattice(
                PermutationGroupRequest(degree=self.degree, generators=self.generators)
            )
        )
        actual_lattice = tuple(
            (entry.group.generators, entry.order) for entry in self.subgroups
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
