"""Typed wire contracts for finite group operations."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel, canonicalize_json_containers

MAX_GROUP_DEGREE = 64

# Conjugacy classes materialize every group element as an array form
# (``|G|`` permutations of length ``degree``).  The degree cap alone does
# not bound this enumeration: S12 already has 479M elements.  Use a
# conservative group-order bound derived with Schreier-Sims before enumeration.
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


def _validation_error(
    code: str, message: str, **context: object
) -> PydanticCustomError:
    return PydanticCustomError(code, message, context)


class PermutationGroup(StrictModel):
    """A finite permutation group given by generator permutations on {0,...,n-1}."""

    degree: int = Field(ge=1, le=MAX_GROUP_DEGREE)
    generators: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_GROUP_DEGREE
    )

    @model_validator(mode="after")
    def require_valid_generators(self) -> Self:
        for perm in self.generators:
            if len(perm) != self.degree:
                raise _validation_error(
                    "group.generator_length",
                    "each generator must have length equal to degree",
                )
            if sorted(perm) != list(range(self.degree)):
                raise _validation_error(
                    "group.generator_permutation",
                    "each generator must be a permutation of 0..n-1",
                )
        return self


class GroupOrderResult(StrictModel):
    """The exact order of a finite permutation group."""

    order: CanonicalInteger


class GroupElementOrderRequest(StrictModel):
    """One generator (or group element) whose order is requested."""

    degree: int = Field(ge=1, le=MAX_GROUP_DEGREE)
    generator: tuple[int, ...] = Field(min_length=1, max_length=MAX_GROUP_DEGREE)


class GroupElementOrderResult(StrictModel):
    """The exact order of one permutation."""

    order: CanonicalInteger


class GroupOrbitRequest(StrictModel):
    """Request the orbit of a point under a permutation group.

    The group is the canonical permutation-group value, so a stabilizer
    result's ``stabilizer`` subgroup feeds the advertised orbit consumer
    unchanged instead of being unpacked into parallel top-level fields.
    """

    group: PermutationGroup = Field(
        description=(
            "Permutation group acting on {0,...,degree-1} as the canonical "
            "value; pass a previous stabilizer or order request's group "
            "value unchanged."
        )
    )
    point: int = Field(ge=0, le=MAX_GROUP_DEGREE - 1)


class GroupOrbitResult(StrictModel):
    """The orbit of a point under a permutation group."""

    orbit: tuple[int, ...] = Field(min_length=1, max_length=MAX_GROUP_DEGREE)
    point: int = Field(ge=0, le=MAX_GROUP_DEGREE - 1)


class GroupConjugacyClassesRequest(StrictModel):
    """Request the conjugacy classes of a bounded permutation group."""

    degree: int = Field(
        ge=1,
        le=MAX_GROUP_DEGREE,
        description=(
            "Degree n of the permutation group acting on {0,...,n-1}; "
            "the generated group must have order at most "
            f"{MAX_CONJUGACY_CLASSES_GROUP_ORDER} (degree "
            f"{MAX_GROUP_DEGREE} alone does not bound enumeration)."
        ),
    )
    generators: tuple[tuple[int, ...], ...] = Field(
        min_length=1,
        max_length=MAX_GROUP_DEGREE,
        description=(
            "Generator permutations as array forms of length degree; "
            "the generated group must have order at most "
            f"{MAX_CONJUGACY_CLASSES_GROUP_ORDER}."
        ),
    )


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
    identically. The producing operation establishes group closure and the
    conjugacy-orbit postcondition; parsing retains only canonical structure.
    """

    classes: tuple[ConjugacyClass, ...] = Field(
        min_length=1,
        max_length=MAX_CONJUGACY_CLASSES_GROUP_ORDER,
    )

    @model_validator(mode="after")
    def require_conjugacy_class_partition(self) -> Self:
        _require_canonical_partition(self.classes)
        return self

    @classmethod
    def _from_kernel(cls, classes: tuple[ConjugacyClass, ...]) -> Self:
        return cls.model_construct(classes=classes)


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
                raise _validation_error(
                    "group.common_degree",
                    "all permutations must have one common degree",
                )
            if sorted(perm) != expected_range:
                raise _validation_error(
                    "group.generator_permutation",
                    "every element must be a permutation of 0..degree-1",
                )
            if perm in elements:
                raise _validation_error(
                    "group.duplicate_element",
                    "conjugacy classes must not repeat an element",
                )
            elements.add(perm)
            if previous_member is not None and perm <= previous_member:
                raise _validation_error(
                    "group.class_order", "class members must be strictly increasing"
                )
            previous_member = perm
        representative = cls[0]
        if (
            previous_representative is not None
            and representative <= previous_representative
        ):
            raise _validation_error(
                "group.class_order",
                "classes must be ordered by canonical representative",
            )
        previous_representative = representative
    if len(elements) > MAX_CONJUGACY_CLASSES_GROUP_ORDER:
        raise _validation_error(
            "group.partition_bound",
            f"a conjugacy-class partition may contain at most "
            f"{MAX_CONJUGACY_CLASSES_GROUP_ORDER} elements",
        )
    return elements


class GroupStabilizerRequest(StrictModel):
    """Request the stabilizer of a point in a permutation group.

    The group is the canonical permutation-group value, so a stabilizer chain
    passes a previous result's ``stabilizer`` subgroup unchanged as ``group``.
    """

    group: PermutationGroup = Field(
        description=(
            "Permutation group acting on {0,...,degree-1} as the canonical "
            "value; pass a previous stabilizer result's `stabilizer` subgroup "
            "unchanged to continue a stabilizer chain."
        )
    )
    point: int = Field(ge=0, le=MAX_GROUP_DEGREE - 1)


def _require_permutation(perm: tuple[int, ...], degree: int, label: str) -> None:
    if len(perm) != degree:
        raise _validation_error(
            "group.generator_length", f"each {label} must have length equal to degree"
        )
    if sorted(perm) != list(range(degree)):
        raise _validation_error(
            "group.generator_permutation",
            f"each {label} must be a permutation of 0..n-1",
        )


class GroupStabilizerResult(StrictModel):
    """Point stabilizer as a canonical permutation-group value bound to its source.

    The result retains the source group and the stabilizer subgroup as nested
    canonical :class:`PermutationGroup` values (``degree`` + ``generators``)
    so the stabilizer can be passed unchanged to ``group.order.compute`` or any
    other permutation-group consumer without reshaping. The trivial stabilizer
    is represented by the identity permutation ``[0,...,degree-1]`` (the
    consumer requires at least one generator). Defining-invariant replay lives
    in the explicit group-owner verifier rather than result deserialization.
    """

    point: int = Field(
        ge=0,
        le=MAX_GROUP_DEGREE - 1,
        description="Point whose stabilizer is computed; must satisfy 0 <= point < source.degree == stabilizer.degree.",
    )
    source: PermutationGroup = Field(
        description="Source permutation group as the canonical value (degree + generators)."
    )
    stabilizer: PermutationGroup = Field(
        description=(
            "Stabilizer subgroup as a canonical permutation-group value on the same "
            "degree; trivial stabilizer is represented by the identity permutation "
            "[0,...,degree-1] so the value is always consumable by group.order.compute "
            "without reshaping or synthesizing an identity."
        )
    )

    @model_validator(mode="after")
    def require_valid_stabilizer(self) -> Self:
        if self.source.degree != self.stabilizer.degree:
            raise _validation_error(
                "group.common_degree", "source and stabilizer must have the same degree"
            )
        if not 0 <= self.point < self.source.degree:
            raise _validation_error(
                "group.point_out_of_range", "point must be in 0..degree-1"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        point: int,
        source: PermutationGroup,
        stabilizer: PermutationGroup,
    ) -> Self:
        """Construct a stabilizer result emitted by the owner-local kernel."""

        return cls.model_construct(point=point, source=source, stabilizer=stabilizer)


class GroupSubgroupLatticeRequest(StrictModel):
    """Enumerate all subgroups of a bounded permutation group.

    The subgroup traversal is exponential in the generated group's order,
    so the lattice is bounded to groups of order at most 64.
    """

    degree: int = Field(ge=1, le=MAX_GROUP_DEGREE)
    generators: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_GROUP_DEGREE
    )


class SubgroupEntry(StrictModel):
    """One subgroup as the canonical permutation-group value plus its order.

    Carrying ``group: PermutationGroup`` lets callers chain an
    enumerated subgroup into ``group.order.compute``, ``group.orbit.compute``,
    or any other permutation-group consumer unchanged.
    """

    group: PermutationGroup
    order: int = Field(ge=1, le=MAX_SUBGROUP_LATTICE_GROUP_ORDER)


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
    detail: str | None = None

    @model_validator(mode="before")
    @classmethod
    def require_bounded_subgroup_entries(cls, data: Any) -> Any:
        data = canonicalize_json_containers(data)
        # Cap entries before nested model construction so a forged relayed
        # payload remains inside the operation-derived transport envelope.
        if isinstance(data, dict):
            entries = data.get("subgroups")
            if isinstance(entries, (list, tuple)) and len(entries) > (
                MAX_SUBGROUP_LATTICE_ENTRIES
            ):
                raise _validation_error(
                    "group.lattice_entry_bound",
                    "subgroups exceed the admitted lattice bound of "
                    f"{MAX_SUBGROUP_LATTICE_ENTRIES} entries (the extremal "
                    "subgroup count among groups of order at most 64)",
                )
        return data

    @model_validator(mode="after")
    def require_outcome_shape(self) -> Self:
        for generator in self.generators:
            _require_permutation(generator, self.degree, "source generator")
        if self.outcome == "COMPUTED":
            if self.subgroups is None or self.detail is not None:
                raise _validation_error(
                    "group.outcome_shape",
                    "computed lattice requires subgroup entries and no detail",
                )
            if self.subgroup_count != len(self.subgroups) or self.subgroup_count < 1:
                raise _validation_error(
                    "group.outcome_shape",
                    "subgroup_count must match the number of subgroups",
                )
        elif self.subgroups is not None or self.detail is None:
            raise _validation_error(
                "group.outcome_shape",
                "an exceeded traversal carries only a safe detail",
            )
        elif self.subgroup_count != 0:
            # No entries are retained on exhaustion, so any positive count
            # would fabricate an exact conclusion.
            raise _validation_error(
                "group.outcome_shape", "an exceeded traversal reports no subgroup count"
            )
        return self

    @classmethod
    def _computed_from_kernel(
        cls,
        request: GroupSubgroupLatticeRequest,
        subgroups: tuple[SubgroupEntry, ...],
    ) -> Self:
        """Construct a complete lattice result emitted by the owner kernel."""

        return cls(
            degree=request.degree,
            generators=request.generators,
            subgroups=subgroups,
            subgroup_count=len(subgroups),
        )

    @classmethod
    def _limit_exceeded_from_kernel(
        cls,
        request: GroupSubgroupLatticeRequest,
        detail: str,
    ) -> Self:
        """Construct the admitted traversal-exhaustion outcome."""

        return cls(
            outcome="LIMIT_EXCEEDED",
            degree=request.degree,
            generators=request.generators,
            detail=detail,
        )
