"""Typed contracts for exact finite cellular sheaves on simplicial complexes.

A cellular sheaf assigns one finite-dimensional based exact vector space to
every nonempty simplex and one linear restriction map ``rho: F(sigma) ->
F(tau)`` to every face inclusion, so that identities hold and saturated-chain
compositions are path independent.  This module owns the request and result
contracts for :func:`from_cover_maps`, which accepts a complete candidate
cover diagram and either returns the canonical sheaf with every derived
comparable-face restriction or rejects the first missing, wrong-axis, or
noncommuting entry as a typed mathematical negative.
"""

from __future__ import annotations

from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian._models import StrictModel
from jacobian.math.topology._models import FiniteSimplicialComplex, Simplex

MAX_SHEAF_SIMPLICES = 64
MAX_SHEAF_STALK_RANK = 8
MAX_SHEAF_TOTAL_STALK_RANK = 512
MAX_SHEAF_COVER_MAPS = 512
MAX_SHEAF_ENTRY_DIGITS = 64
MAX_SHEAF_PRIME = 1000003
MAX_SHEAF_DERIVED_RESTRICTIONS = 2048
MAX_SHEAF_RESTRICTION_CELLS = 65536
MAX_SHEAF_DIAMONDS = 65536
MAX_SHEAF_COHOMOLOGY_CELLS = 65536
MAX_SHEAF_HODGE_MATRIX_CELLS = 65536
MAX_SHEAF_HODGE_CUBIC_WORK = 4000000
MAX_SHEAF_HODGE_RESULT_DIGIT_WORK = 8000000
MAX_SHEAF_SECTION_ROWS = (
    MAX_SHEAF_SIMPLICES * (MAX_SHEAF_SIMPLICES - 1) // 2 * MAX_SHEAF_STALK_RANK
)
MAX_SHEAF_SECTION_MATRIX_CELLS = 65536
MAX_SHEAF_SECTION_OUTPUT_CELLS = 200000
MAX_SHEAF_SECTION_WORK = 4000000
MAX_SHEAF_SECTION_RESULT_DIGIT_WORK = 8000000
MAX_SHEAF_SECTION_RESTRICTION_DIGIT_WORK = 34000000
MAX_SHEAF_MORPHISM_COMPONENT_CELLS = 32768
MAX_SHEAF_MORPHISM_WORK = 4000000
MAX_SHEAF_MORPHISM_RESULT_DIGIT_WORK = 8000000
MAX_SHEAF_RESTRICTION_RESULT_DIGIT_WORK = 8000000

BasisLabel = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,31}$",
        strict=True,
    ),
]

SheafScalar = CanonicalRational | StrictInt


def sheaf_scalar_digits(value: SheafScalar) -> int:
    if isinstance(value, CanonicalRational):
        return canonical_rational_component_digits(value)
    if type(value) is int and value.bit_length() <= 215:
        return len(str(abs(value)))
    return MAX_SHEAF_ENTRY_DIGITS + 1


def sheaf_scalar_digit_work(count: int, digits: int = MAX_SHEAF_ENTRY_DIGITS) -> int:
    """Bound exact scalar component digits across a collection of values."""
    return count * digits


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"topology.cellular_sheaf.{reason}", message)


def _require_canonical_simplex(simplex: Simplex, *, label: str) -> None:
    if not simplex:
        raise _validation_error("empty_simplex", f"{label} must be a nonempty simplex")
    if tuple(sorted(simplex)) != simplex:
        raise _validation_error(
            "simplex_not_canonical",
            f"{label} must use canonical increasing vertex order",
        )
    if len(set(simplex)) != len(simplex):
        raise _validation_error(
            "simplex_not_canonical", f"{label} must have distinct vertices"
        )


def _require_field_scalars(
    matrices: tuple[tuple[tuple[SheafScalar, ...], ...], ...],
    coefficient_field: SheafField,
    prime: int | None,
    *,
    label: str,
) -> None:
    for matrix in matrices:
        for row in matrix:
            for scalar in row:
                if coefficient_field is SheafField.RATIONAL:
                    valid = isinstance(scalar, CanonicalRational)
                else:
                    valid = type(scalar) is int and (
                        prime is None or 0 <= scalar < prime
                    )
                if not valid:
                    raise _validation_error(
                        "scalar_parent_mismatch",
                        f"{label} scalars must use the declared coefficient field's canonical type",
                    )


class SheafField(StrEnum):
    """Exact coefficient fields supported by cellular sheaves."""

    RATIONAL = "QQ"
    PRIME_FIELD = "GF_p"


class SheafOutcome(StrEnum):
    """Discriminated outcome of one cover-diagram construction."""

    CELLULAR_SHEAF = "CELLULAR_SHEAF"
    NOT_A_SHEAF = "NOT_A_SHEAF"


class SheafObstructionCode(StrEnum):
    """Stable codes for the first diagram obstruction."""

    MISSING_COVER_MAP = "MISSING_COVER_MAP"
    COVER_MAP_WRONG_AXIS = "COVER_MAP_WRONG_AXIS"
    NONCOMMUTING_DIAMOND = "NONCOMMUTING_DIAMOND"


class SheafStalk(StrictModel):
    """A based finite-dimensional stalk on one nonempty simplex.

    ``basis`` is an ordered tuple of unique basis labels; its length is the
    stalk rank.  The zero-dimensional stalk is represented by an empty basis
    and is a first-class value.
    """

    simplex: Simplex
    basis: tuple[BasisLabel, ...] = Field(default=())

    @model_validator(mode="after")
    def require_based_stalk(self) -> Self:
        _require_canonical_simplex(self.simplex, label="stalk simplex")
        if len(set(self.basis)) != len(self.basis):
            raise _validation_error(
                "basis_labels_not_unique",
                "stalk basis labels must be unique",
            )
        return self


class CoverRestrictionMatrix(StrictModel):
    """One candidate restriction matrix on a face-poset cover inclusion.

    ``source`` must be a codimension-one face of ``target``.  ``entries`` is
    a dense row-major matrix whose rows index the target stalk basis and whose
    columns index the source stalk basis, matching ``rho: F(source) ->
    F(target)``. Entries are exact field values: ``CanonicalRational`` over
    ``QQ`` and strict integer representatives over ``GF(p)``.
    """

    source: Simplex
    target: Simplex
    entries: tuple[tuple[SheafScalar, ...], ...] = Field(default=())

    @model_validator(mode="after")
    def require_rectangular_matrix(self) -> Self:
        _require_canonical_simplex(self.source, label="cover source")
        _require_canonical_simplex(self.target, label="cover target")
        widths = {len(row) for row in self.entries}
        if len(widths) > 1:
            raise _validation_error(
                "matrix_not_rectangular",
                "cover restriction entries must be a rectangular matrix",
            )
        return self


class FromCoverMapsRequest(StrictModel):
    """Build a cellular sheaf from a complete candidate cover diagram."""

    complex: FiniteSimplicialComplex
    coefficient_field: SheafField = SheafField.RATIONAL
    prime: StrictInt | None = Field(default=None, ge=2, le=MAX_SHEAF_PRIME)
    stalks: tuple[SheafStalk, ...] = Field(default=())
    cover_maps: tuple[CoverRestrictionMatrix, ...] = Field(default=())

    @model_validator(mode="after")
    def require_cover_scalar_parent(self) -> Self:
        _require_field_scalars(
            tuple(item.entries for item in self.cover_maps),
            self.coefficient_field,
            self.prime,
            label="cover map",
        )
        return self


class SheafRestriction(StrictModel):
    """One exact restriction ``rho: F(source) -> F(target)``.

    Rows are indexed by ``row_basis`` (the target stalk basis) and columns by
    ``column_basis`` (the source stalk basis).  ``cover_path`` records the
    saturated chain of simplices along which a derived composite was taken;
    for a cover inclusion it is exactly ``(source, target)``.
    """

    source: Simplex
    target: Simplex
    row_basis: tuple[BasisLabel, ...] = Field(default=())
    column_basis: tuple[BasisLabel, ...] = Field(default=())
    entries: tuple[tuple[SheafScalar, ...], ...] = Field(default=())
    cover_path: tuple[Simplex, ...] = Field(default=())

    @model_validator(mode="after")
    def require_bound_matrix(self) -> Self:
        _require_canonical_simplex(self.source, label="restriction source")
        _require_canonical_simplex(self.target, label="restriction target")
        if len(self.entries) != len(self.row_basis) or any(
            len(row) != len(self.column_basis) for row in self.entries
        ):
            raise _validation_error(
                "restriction_axis_mismatch",
                "restriction entries must match the declared row and column bases",
            )
        if self.cover_path:
            if self.cover_path[0] != self.source or self.cover_path[-1] != self.target:
                raise _validation_error(
                    "cover_path_endpoints",
                    "cover path must start at the source and end at the target",
                )
            for earlier, later in pairwise(self.cover_path):
                if not set(earlier) < set(later) or len(later) != len(earlier) + 1:
                    raise _validation_error(
                        "cover_path_not_saturated",
                        "cover path steps must be codimension-one face inclusions",
                    )
        return self


class FiniteCellularSheaf(StrictModel):
    """A checked cellular sheaf with its complete derived restriction diagram.

    ``stalks`` binds one based stalk to every nonempty simplex in canonical
    order.  ``cover_restrictions`` covers every codimension-one inclusion and
    ``derived_restrictions`` every longer comparable inclusion.  ``diamonds``
    counts the elementary commuting squares that were replayed and
    ``comparable_pairs`` counts every ordered comparable face pair; both are
    inspectable accounting, never re-derived by a validator.
    """

    complex: FiniteSimplicialComplex
    coefficient_field: SheafField
    prime: StrictInt | None = Field(default=None, ge=2, le=MAX_SHEAF_PRIME)
    stalks: tuple[SheafStalk, ...] = Field(default=())
    cover_restrictions: tuple[SheafRestriction, ...] = Field(default=())
    derived_restrictions: tuple[SheafRestriction, ...] = Field(default=())
    diamonds: StrictInt = Field(default=0, ge=0)
    comparable_pairs: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def require_bound_diagram(self) -> Self:
        if self.coefficient_field is SheafField.PRIME_FIELD:
            if self.prime is None:
                raise _validation_error(
                    "prime_required", "GF(p) sheaves must carry their prime modulus"
                )
        elif self.prime is not None:
            raise _validation_error(
                "prime_forbidden", "QQ sheaves must not carry a prime modulus"
            )
        canonical = self.canonical_face_order
        if tuple(stalk.simplex for stalk in self.stalks) != canonical:
            raise _validation_error(
                "stalk_order_invalid",
                "stalks must bind every face in canonical dimension/lexical order",
            )
        basis_for = {stalk.simplex: stalk.basis for stalk in self.stalks}
        seen_pairs: set[tuple[Simplex, Simplex]] = set()
        for restriction in (
            *self.cover_restrictions,
            *self.derived_restrictions,
        ):
            if (
                basis_for.get(restriction.target) != restriction.row_basis
                or basis_for.get(restriction.source) != restriction.column_basis
            ):
                raise _validation_error(
                    "restriction_basis_unbound",
                    "restriction axes must equal the declared stalk bases",
                )
            _require_field_scalars(
                (restriction.entries,),
                self.coefficient_field,
                self.prime,
                label="restriction",
            )
            key = (restriction.source, restriction.target)
            if key in seen_pairs:
                raise _validation_error(
                    "duplicate_restriction",
                    "each comparable face pair carries exactly one restriction",
                )
            seen_pairs.add(key)
        return self

    @property
    def canonical_face_order(self) -> tuple[Simplex, ...]:
        return tuple(
            face for group in self.complex.faces_by_dimension for face in group.faces
        )

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class SheafSubcomplexRequest(StrictModel):
    """Restrict a checked cellular sheaf to an included subcomplex."""

    sheaf: FiniteCellularSheaf
    subcomplex: FiniteSimplicialComplex


class SheafSubcomplexResult(StrictModel):
    """The source sheaf and its restriction to an included subcomplex."""

    source: FiniteCellularSheaf
    subcomplex: FiniteCellularSheaf

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class DiamondCounterexample(StrictModel):
    """Two saturated cover paths with unequal composed restriction matrices.

    Both paths run from ``source`` to ``target`` through distinct intermediate
    cells; the matrices are the exact composites in the declared field's
    canonical scalar type.
    """

    source: Simplex
    target: Simplex
    first_path: tuple[Simplex, ...]
    second_path: tuple[Simplex, ...]
    first_matrix: tuple[tuple[SheafScalar, ...], ...]
    second_matrix: tuple[tuple[SheafScalar, ...], ...]

    @model_validator(mode="after")
    def require_parallel_paths(self) -> Self:
        for path in (self.first_path, self.second_path):
            if len(path) < 3 or path[0] != self.source or path[-1] != self.target:
                raise _validation_error(
                    "diamond_path_endpoints",
                    "diamond paths must run from the source to the target",
                )
        if self.first_path == self.second_path:
            raise _validation_error(
                "diamond_paths_identical",
                "a noncommuting diamond requires two distinct paths",
            )
        if tuple(len(row) for row in self.first_matrix) != tuple(
            len(row) for row in self.second_matrix
        ) or len(self.first_matrix) != len(self.second_matrix):
            raise _validation_error(
                "diamond_matrix_shape",
                "both composites must share the source/target stalk shape",
            )
        return self


class SheafObstruction(StrictModel):
    """The first obstruction that keeps a candidate diagram from being a sheaf."""

    code: SheafObstructionCode
    message: str = Field(min_length=1, max_length=512)
    source: Simplex | None = None
    target: Simplex | None = None
    diamond: DiamondCounterexample | None = None

    @model_validator(mode="after")
    def require_obstruction_payload(self) -> Self:
        if self.code is SheafObstructionCode.NONCOMMUTING_DIAMOND:
            if self.diamond is None:
                raise _validation_error(
                    "diamond_required",
                    "a noncommuting obstruction requires both concrete paths",
                )
        elif self.diamond is not None:
            raise _validation_error(
                "diamond_forbidden",
                "only a noncommuting obstruction carries a diamond",
            )
        if self.code is not SheafObstructionCode.NONCOMMUTING_DIAMOND and (
            self.source is None or self.target is None
        ):
            raise _validation_error(
                "obstruction_pair_required",
                "a missing or wrong-axis obstruction names its cover pair",
            )
        return self


class FromCoverMapsResult(StrictModel):
    """Discriminated construction outcome bound to its source diagram."""

    outcome: SheafOutcome
    sheaf: FiniteCellularSheaf | None = None
    obstruction: SheafObstruction | None = None

    @model_validator(mode="after")
    def require_discriminated_payload(self) -> Self:
        if self.outcome is SheafOutcome.CELLULAR_SHEAF:
            if self.sheaf is None or self.obstruction is not None:
                raise _validation_error(
                    "sheaf_payload_conflict",
                    "a sheaf outcome carries the canonical sheaf only",
                )
        elif self.sheaf is not None or self.obstruction is None:
            raise _validation_error(
                "obstruction_payload_conflict",
                "a negative outcome carries its obstruction only",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class SheafCohomologyRequest(StrictModel):
    """Compute cellular cohomology of a checked finite cellular sheaf."""

    sheaf: FiniteCellularSheaf


class SheafCochainCoordinate(StrictModel):
    """One cochain basis vector: one basis label of one stalk."""

    simplex: Simplex
    basis_label: BasisLabel

    @model_validator(mode="after")
    def require_nonempty_simplex(self) -> Self:
        _require_canonical_simplex(self.simplex, label="cochain simplex")
        return self


class SheafSectionCompatibilityAxis(StrictModel):
    """One equation row for a target-stalk coordinate of a face inclusion."""

    source: Simplex
    target: Simplex
    target_basis_label: BasisLabel

    @model_validator(mode="after")
    def require_face_inclusion(self) -> Self:
        _require_canonical_simplex(self.source, label="compatibility source")
        _require_canonical_simplex(self.target, label="compatibility target")
        if not set(self.source) < set(self.target):
            raise _validation_error(
                "section_compatibility_axis",
                "compatibility rows must bind a strict face inclusion",
            )
        return self


class SheafSectionEvaluation(StrictModel):
    """Projection from section coordinates into one based stalk."""

    simplex: Simplex
    stalk_basis: tuple[BasisLabel, ...] = Field(max_length=MAX_SHEAF_STALK_RANK)
    section_basis: tuple[BasisLabel, ...] = Field(max_length=MAX_SHEAF_TOTAL_STALK_RANK)
    entries: tuple[tuple[SheafScalar, ...], ...] = Field(
        max_length=MAX_SHEAF_STALK_RANK
    )

    @model_validator(mode="after")
    def require_evaluation_shape(self) -> Self:
        _require_canonical_simplex(self.simplex, label="evaluation simplex")
        if len(self.entries) != len(self.stalk_basis) or any(
            len(row) != len(self.section_basis) for row in self.entries
        ):
            raise _validation_error(
                "section_evaluation_axis",
                "evaluation entries must map the section basis into the stalk basis",
            )
        return self


class SheafSectionSpace(StrictModel):
    """Exact source-bound vector space of compatible stalk assignments.

    The dense compatibility matrix has one row for every target-stalk basis
    coordinate of every strict comparable face pair. Its columns follow
    ``ambient_basis``. ``basis_coordinates`` stores nullspace vectors as rows,
    and each stalk evaluation matrix maps those section-basis coordinates into
    the declared stalk basis.
    """

    sheaf: FiniteCellularSheaf
    dimension: StrictInt = Field(ge=0, le=MAX_SHEAF_TOTAL_STALK_RANK)
    section_basis: tuple[BasisLabel, ...] = Field(max_length=MAX_SHEAF_TOTAL_STALK_RANK)
    ambient_basis: tuple[SheafCochainCoordinate, ...] = Field(
        max_length=MAX_SHEAF_TOTAL_STALK_RANK
    )
    compatibility_row_axes: tuple[SheafSectionCompatibilityAxis, ...] = Field(
        max_length=MAX_SHEAF_SECTION_ROWS
    )
    compatibility_matrix: tuple[tuple[SheafScalar, ...], ...] = Field(
        max_length=MAX_SHEAF_SECTION_ROWS
    )
    basis_coordinates: tuple[tuple[SheafScalar, ...], ...] = Field(
        max_length=MAX_SHEAF_TOTAL_STALK_RANK
    )
    evaluations: tuple[SheafSectionEvaluation, ...] = Field(
        max_length=MAX_SHEAF_SIMPLICES
    )

    @model_validator(mode="after")
    def require_section_space_axes(self) -> Self:
        if len(set(self.section_basis)) != len(self.section_basis):
            raise _validation_error(
                "section_basis_not_unique", "section basis labels must be unique"
            )
        if self.dimension != len(self.section_basis) or self.dimension != len(
            self.basis_coordinates
        ):
            raise _validation_error(
                "section_dimension_mismatch",
                "dimension must match the section basis and coordinate vectors",
            )
        expected_ambient = tuple(
            SheafCochainCoordinate(simplex=stalk.simplex, basis_label=label)
            for stalk in self.sheaf.stalks
            for label in stalk.basis
        )
        if self.ambient_basis != expected_ambient:
            raise _validation_error(
                "section_ambient_axis",
                "ambient basis must follow the source sheaf's canonical stalk axes",
            )
        ambient_dimension = len(self.ambient_basis)
        if any(len(vector) != ambient_dimension for vector in self.basis_coordinates):
            raise _validation_error(
                "section_basis_coordinates_axis",
                "each section basis vector must use the complete ambient stalk axis",
            )
        if len(self.compatibility_matrix) != len(self.compatibility_row_axes) or any(
            len(row) != ambient_dimension for row in self.compatibility_matrix
        ):
            raise _validation_error(
                "section_compatibility_matrix_axis",
                "compatibility matrix rows must match their labelled ambient axes",
            )
        expected_axes = tuple(
            (source, target, label)
            for source in self.sheaf.canonical_face_order
            for target in self.sheaf.canonical_face_order
            if set(source) < set(target)
            for label in next(
                stalk.basis for stalk in self.sheaf.stalks if stalk.simplex == target
            )
        )
        actual_axes = tuple(
            (axis.source, axis.target, axis.target_basis_label)
            for axis in self.compatibility_row_axes
        )
        if actual_axes != expected_axes:
            raise _validation_error(
                "section_compatibility_row_axes",
                "compatibility rows must cover every comparable pair in canonical order",
            )
        if tuple(item.simplex for item in self.evaluations) != tuple(
            stalk.simplex for stalk in self.sheaf.stalks
        ):
            raise _validation_error(
                "section_evaluation_coverage",
                "evaluations must cover every source stalk in canonical order",
            )
        for evaluation, stalk in zip(self.evaluations, self.sheaf.stalks, strict=True):
            if (
                evaluation.stalk_basis != stalk.basis
                or evaluation.section_basis != self.section_basis
            ):
                raise _validation_error(
                    "section_evaluation_context",
                    "each evaluation must retain its source stalk and section bases",
                )
            _require_field_scalars(
                (evaluation.entries,),
                self.sheaf.coefficient_field,
                self.sheaf.prime,
                label="section evaluation",
            )
        _require_field_scalars(
            (self.compatibility_matrix, self.basis_coordinates),
            self.sheaf.coefficient_field,
            self.sheaf.prime,
            label="section space",
        )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class SheafSectionRestriction(StrictModel):
    """The induced linear map on sections for one included subcomplex."""

    source: SheafSectionSpace
    target: SheafSectionSpace
    entries: tuple[tuple[SheafScalar, ...], ...] = Field(
        max_length=MAX_SHEAF_TOTAL_STALK_RANK
    )

    @model_validator(mode="after")
    def require_restriction_axes(self) -> Self:
        if len(self.entries) != self.target.dimension or any(
            len(row) != self.source.dimension for row in self.entries
        ):
            raise _validation_error(
                "section_restriction_axis",
                "section restriction entries must map the source section basis to the target section basis",
            )
        if (
            sheaf_scalar_digit_work(
                sum(len(row) for row in self.entries),
                max(
                    (
                        sheaf_scalar_digits(value)
                        for row in self.entries
                        for value in row
                    ),
                    default=1,
                ),
            )
            > MAX_SHEAF_SECTION_RESTRICTION_DIGIT_WORK
        ):
            raise _validation_error(
                "section_restriction_output",
                "section restriction scalar values exceed their declared digit-work bound",
            )
        source_cells = set(self.source.sheaf.canonical_face_order)
        target_cells = set(self.target.sheaf.canonical_face_order)
        if not target_cells.issubset(source_cells):
            raise _validation_error(
                "section_restriction_parent",
                "the target section sheaf must be a subcomplex restriction of the source",
            )
        if (self.source.sheaf.coefficient_field, self.source.sheaf.prime) != (
            self.target.sheaf.coefficient_field,
            self.target.sheaf.prime,
        ):
            raise _validation_error(
                "section_restriction_field", "section restriction fields must agree"
            )
        _require_field_scalars(
            (self.entries,),
            self.source.sheaf.coefficient_field,
            self.source.sheaf.prime,
            label="section restriction",
        )
        source_stalks = {stalk.simplex: stalk for stalk in self.source.sheaf.stalks}
        if any(
            source_stalks.get(stalk.simplex) != stalk
            for stalk in self.target.sheaf.stalks
        ):
            raise _validation_error(
                "section_restriction_stalks",
                "target stalk axes must be retained exactly from the source sheaf",
            )
        source_maps = {
            (item.source, item.target): item
            for item in (
                *self.source.sheaf.cover_restrictions,
                *self.source.sheaf.derived_restrictions,
            )
        }
        target_maps = (
            *self.target.sheaf.cover_restrictions,
            *self.target.sheaf.derived_restrictions,
        )
        if any(
            source_maps.get((item.source, item.target)) != item for item in target_maps
        ):
            raise _validation_error(
                "section_restriction_maps",
                "target restriction maps must be retained exactly from the source sheaf",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class SheafCohomologyGroup(StrictModel):
    """Cohomology in one degree with representative cocycles.

    ``cocycle_representatives`` holds one cochain per cohomology basis
    vector in ``C^k`` coordinates; its length is the Betti number.
    """

    degree: StrictInt = Field(ge=0)
    cochain_dimension: StrictInt = Field(ge=0)
    cocycle_rank: StrictInt = Field(ge=0)
    coboundary_rank: StrictInt = Field(ge=0)
    betti_number: StrictInt = Field(ge=0)
    cocycle_representatives: tuple[tuple[SheafScalar, ...], ...] = Field(default=())


class SheafCoboundaryLedgerEntry(StrictModel):
    """One replayed delta^{k+1} delta^k = 0 product."""

    degree: StrictInt = Field(ge=0)
    product_rows: StrictInt = Field(ge=0)
    product_columns: StrictInt = Field(ge=0)
    nonzero_entries: Literal[0] = 0


class SheafCohomologyResult(StrictModel):
    """Cellular cohomology of a retained finite cellular sheaf.

    ``coboundary_matrices[k]`` is the dense signed-incidence coboundary
    ``C^k -> C^{k+1}`` (rows index the degree-``k+1`` cochain basis,
    columns the degree-``k`` basis) and ``cochain_bases[k]`` binds each
    axis to its stalk coordinates. ``euler_characteristic_stalk`` is the
    alternating stalk-rank sum and ``euler_characteristic_cohomology``
    the alternating Betti sum; both are stored kernel accounting whose
    equality the kernel establishes for finite free stalks.
    """

    sheaf: FiniteCellularSheaf
    cochain_dimensions: tuple[StrictInt, ...] = Field(min_length=1)
    cochain_bases: tuple[tuple[SheafCochainCoordinate, ...], ...] = Field(min_length=1)
    coboundary_matrices: tuple[tuple[tuple[SheafScalar, ...], ...], ...] = Field(
        default=()
    )
    groups: tuple[SheafCohomologyGroup, ...] = Field(min_length=1)
    euler_characteristic_stalk: StrictInt
    euler_characteristic_cohomology: StrictInt
    differential_squared_zero: tuple[SheafCoboundaryLedgerEntry, ...] = Field(
        default=()
    )

    @model_validator(mode="after")
    def require_structural_cohomology(self) -> Self:  # noqa: C901
        dimension = self.sheaf.complex.dimension
        if len(self.cochain_dimensions) != dimension + 1:
            raise _validation_error(
                "cohomology_degree_coverage_invalid",
                "cochain dimensions must cover every simplex dimension",
            )
        if len(self.cochain_bases) != dimension + 1 or any(
            len(basis) != size
            for basis, size in zip(
                self.cochain_bases, self.cochain_dimensions, strict=True
            )
        ):
            raise _validation_error(
                "cochain_basis_axis_mismatch",
                "cochain bases must match the cochain dimensions",
            )
        faces = {
            face
            for group in self.sheaf.complex.faces_by_dimension
            for face in group.faces
        }
        for degree, basis in enumerate(self.cochain_bases):
            for coordinate in basis:
                if coordinate.simplex not in faces or len(coordinate.simplex) != (
                    degree + 1
                ):
                    raise _validation_error(
                        "cochain_coordinate_face_invalid",
                        "every cochain coordinate must name a face of its degree",
                    )
        if len(self.coboundary_matrices) != dimension:
            raise _validation_error(
                "coboundary_count_mismatch",
                "there must be one coboundary matrix per adjacent degree pair",
            )
        for degree, matrix in enumerate(self.coboundary_matrices):
            rows = self.cochain_dimensions[degree + 1]
            columns = self.cochain_dimensions[degree]
            if len(matrix) != rows or any(len(row) != columns for row in matrix):
                raise _validation_error(
                    "coboundary_shape_mismatch",
                    f"coboundary {degree} must have shape C^{degree + 1} x C^{degree}",
                )
        _require_field_scalars(
            self.coboundary_matrices
            + tuple(group.cocycle_representatives for group in self.groups),
            self.sheaf.coefficient_field,
            self.sheaf.prime,
            label="cohomology result",
        )
        if tuple(group.degree for group in self.groups) != tuple(range(dimension + 1)):
            raise _validation_error(
                "cohomology_degree_coverage_invalid",
                "cohomology groups must cover every degree exactly once",
            )
        for degree, group in enumerate(self.groups):
            if group.cochain_dimension != self.cochain_dimensions[degree]:
                raise _validation_error(
                    "cohomology_cochain_dimension_mismatch",
                    "each group must retain its cochain dimension",
                )
            if group.cocycle_rank > group.cochain_dimension:
                raise _validation_error(
                    "cocycle_rank_exceeded",
                    "cocycle rank cannot exceed the cochain dimension",
                )
            if group.betti_number != group.cocycle_rank - group.coboundary_rank:
                raise _validation_error(
                    "cohomology_rank_identity_invalid",
                    "betti_number must equal cocycle_rank - coboundary_rank",
                )
            if group.betti_number < 0:
                raise _validation_error(
                    "betti_number_negative",
                    "the Betti number cannot be negative",
                )
            if degree == 0 and group.coboundary_rank != 0:
                raise _validation_error(
                    "coboundary_rank_degree_zero_invalid",
                    "nothing bounds into degree zero",
                )
            if len(group.cocycle_representatives) != group.betti_number or any(
                len(vector) != group.cochain_dimension
                for vector in group.cocycle_representatives
            ):
                raise _validation_error(
                    "cocycle_representative_axis_invalid",
                    "cocycle representatives must span the Betti number in "
                    "cochain coordinates",
                )
        if self.euler_characteristic_stalk != self.euler_characteristic_cohomology:
            raise _validation_error(
                "euler_characteristic_mismatch",
                "the stalk and cohomology Euler characteristics must agree",
            )
        if {entry.degree for entry in self.differential_squared_zero} != set(
            range(max(0, dimension - 1))
        ) or len(self.differential_squared_zero) != max(0, dimension - 1):
            raise _validation_error(
                "coboundary_square_ledger_incomplete",
                "the square-zero ledger must cover every adjacent "
                "coboundary pair exactly once",
            )
        for entry in self.differential_squared_zero:
            rows = self.cochain_dimensions[entry.degree + 2]
            columns = self.cochain_dimensions[entry.degree]
            if entry.product_rows != rows or entry.product_columns != columns:
                raise _validation_error(
                    "coboundary_square_product_shape_mismatch",
                    "each ledger product shape must match its outer axes",
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class SheafHodgeRequest(StrictModel):
    """Request exact Hodge operators for standard rational stalk metrics."""

    sheaf: FiniteCellularSheaf


class SheafHodgeResult(StrictModel):
    """Hodge Laplacians and harmonic bases in source-bound cochain axes.

    The standard positive-definite coordinate metric is recorded on every
    stalk by its identity Gram matrix. Cochain metrics are their orthogonal
    direct sums in the retained degreewise cochain bases.
    """

    sheaf: FiniteCellularSheaf
    cochain_bases: tuple[tuple[SheafCochainCoordinate, ...], ...]
    stalk_gram_matrices: tuple[tuple[tuple[SheafScalar, ...], ...], ...]
    up_laplacians: tuple[tuple[tuple[SheafScalar, ...], ...], ...]
    down_laplacians: tuple[tuple[tuple[SheafScalar, ...], ...], ...]
    laplacians: tuple[tuple[tuple[SheafScalar, ...], ...], ...]
    harmonic_bases: tuple[tuple[tuple[SheafScalar, ...], ...], ...]

    @model_validator(mode="after")
    def require_hodge_axes(self) -> Self:
        stalks = self.sheaf.stalks
        if len(self.stalk_gram_matrices) != len(stalks):
            raise _validation_error(
                "hodge_metric_stalk_coverage",
                "one coordinate Gram matrix is required per source stalk",
            )
        for stalk, gram in zip(stalks, self.stalk_gram_matrices, strict=True):
            rank = len(stalk.basis)
            if len(gram) != rank or any(len(row) != rank for row in gram):
                raise _validation_error(
                    "hodge_metric_axis_mismatch",
                    "stalk Gram matrix axes must match their source basis",
                )
            expected_gram = tuple(
                tuple(
                    (
                        CanonicalRational(num=int(i == j), den=1)
                        if self.sheaf.coefficient_field is SheafField.RATIONAL
                        else int(i == j)
                    )
                    for j in range(rank)
                )
                for i in range(rank)
            )
            if gram != expected_gram:
                raise _validation_error(
                    "hodge_metric_not_standard",
                    "this operation supports the standard identity stalk metrics",
                )
        dimensions = tuple(len(axis) for axis in self.cochain_bases)
        if len(dimensions) != self.sheaf.complex.dimension + 1:
            raise _validation_error(
                "hodge_degree_coverage",
                "Hodge axes must cover every source simplex degree",
            )
        if any(
            len(matrices) != len(dimensions)
            for matrices in (
                self.up_laplacians,
                self.down_laplacians,
                self.laplacians,
                self.harmonic_bases,
            )
        ):
            raise _validation_error(
                "hodge_degree_coverage",
                "one Hodge matrix and harmonic basis are required per degree",
            )
        for degree, size in enumerate(dimensions):
            for family, matrices in (
                ("up", self.up_laplacians),
                ("down", self.down_laplacians),
                ("Hodge", self.laplacians),
            ):
                matrix = matrices[degree]
                if len(matrix) != size or any(len(row) != size for row in matrix):
                    raise _validation_error(
                        "hodge_matrix_axis_mismatch",
                        f"the degree-{degree} {family} Laplacian must be square on its cochain axis",
                    )
            basis = self.harmonic_bases[degree]
            if any(len(vector) != size for vector in basis):
                raise _validation_error(
                    "harmonic_basis_axis_mismatch",
                    f"degree-{degree} harmonic vectors must use its cochain axis",
                )
        _require_field_scalars(
            self.stalk_gram_matrices
            + self.up_laplacians
            + self.down_laplacians
            + self.laplacians
            + self.harmonic_bases,
            self.sheaf.coefficient_field,
            self.sheaf.prime,
            label="Hodge result",
        )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


__all__ = [
    "MAX_SHEAF_COHOMOLOGY_CELLS",
    "MAX_SHEAF_COVER_MAPS",
    "MAX_SHEAF_DERIVED_RESTRICTIONS",
    "MAX_SHEAF_DIAMONDS",
    "MAX_SHEAF_ENTRY_DIGITS",
    "MAX_SHEAF_HODGE_CUBIC_WORK",
    "MAX_SHEAF_HODGE_MATRIX_CELLS",
    "MAX_SHEAF_HODGE_RESULT_DIGIT_WORK",
    "MAX_SHEAF_PRIME",
    "MAX_SHEAF_RESTRICTION_CELLS",
    "MAX_SHEAF_SIMPLICES",
    "MAX_SHEAF_STALK_RANK",
    "MAX_SHEAF_TOTAL_STALK_RANK",
    "BasisLabel",
    "CoverRestrictionMatrix",
    "DiamondCounterexample",
    "FiniteCellularSheaf",
    "FromCoverMapsRequest",
    "FromCoverMapsResult",
    "SheafCoboundaryLedgerEntry",
    "SheafCochainCoordinate",
    "SheafCohomologyGroup",
    "SheafCohomologyRequest",
    "SheafCohomologyResult",
    "SheafField",
    "SheafHodgeRequest",
    "SheafHodgeResult",
    "SheafObstruction",
    "SheafObstructionCode",
    "SheafOutcome",
    "SheafRestriction",
    "SheafStalk",
]
