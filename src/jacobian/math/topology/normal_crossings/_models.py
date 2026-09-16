"""Typed wire contracts for simple normal crossings operations.

The owner-local canonical contract for a strict SNC presentation:

- A stratum is identified by its exact component set; duplicate component
  sets are rejected (``topology.normal_crossings.duplicate_stratum``).
- Incidence is downward closed: every nonempty subset of a declared
  stratum's component set is itself a declared stratum, and every
  component carries its singleton stratum.
- Branch multiplicity ``r`` of a stratum is its component-set cardinality.
- Dimensions satisfy the transverse identity ``dim S = dim {i} - (r - 1)``
  for each component ``i`` of ``S``; one added branch drops the dimension
  by exactly one.
- Nearby-cycle lattices use the canonical saturated successive-difference
  basis ``b_k = e_k - e_{k+1}`` of ``K_S = ker(sum: Z^r -> Z)`` in the
  stratum's sorted component (branch) order.
- For a component-set inclusion ``I subset J``, the specialization map is
  ``K_J -> K_I`` (the deeper stratum with more branches specializes to the
  shallower one): the fiber-sum fold acts as the identity on ``I`` and
  folds every branch of ``J \\ I`` onto the largest component of ``I`` in
  sorted order; matrices are expressed in the successive-difference bases.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.topology._models import (
    BoundarySquareLedgerEntry,
    FiniteSimplicialComplex,
    VertexLabel,
)
from jacobian.math.topology.chain_complexes.values import ChainComplexValue

MAX_NC_COMPONENTS = 32
MAX_NC_STRATA = 256
MAX_NC_BRANCH_MULTIPLICITY = 8
MAX_NC_STRATUM_DIMENSION = 64
MAX_NC_SPECIALIZATION_MAPS = 2048
MAX_NC_SPECIALIZATION_CELLS = 131_072
MAX_NC_LATTICE_ENTRY_DIGITS = 8


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by normal-crossings contracts."""

    return PydanticCustomError(f"topology.normal_crossings.{reason}", message)


def _require_sorted_distinct(labels: tuple[VertexLabel, ...]) -> None:
    if tuple(sorted(set(labels))) != labels:
        raise _validation_error(
            "component_order",
            "component tuples must be sorted and distinct in canonical order",
        )


class NormalCrossingsStratum(StrictModel):
    """One connected intersection stratum bound to its exact component set."""

    components: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_NC_BRANCH_MULTIPLICITY,
    )
    dimension: int = Field(ge=0, le=MAX_NC_STRATUM_DIMENSION)

    @model_validator(mode="after")
    def require_canonical_components(self) -> Self:
        _require_sorted_distinct(self.components)
        return self

    @property
    def branch_multiplicity(self) -> int:
        return len(self.components)


class NormalCrossingsPresentation(StrictModel):
    """A finite strict-SNC incidence presentation in canonical order.

    Structural decoding only checks bounded sorted shapes; the semantic
    incidence identities (downward closure, singleton coverage, dimension
    identity) are established once by the shared owner admission.
    """

    components: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_NC_COMPONENTS,
    )
    strata: tuple[NormalCrossingsStratum, ...] = Field(
        min_length=1,
        max_length=MAX_NC_STRATA,
    )

    @model_validator(mode="after")
    def require_canonical_order(self) -> Self:
        _require_sorted_distinct(self.components)
        keys = tuple(stratum.components for stratum in self.strata)
        canonical = tuple(sorted(set(keys), key=lambda key: (len(key), key)))
        if keys != canonical:
            raise _validation_error(
                "stratum_order",
                "strata must be distinct and sorted by cardinality, then by "
                "component set",
            )
        return self


class NormalCrossingsPresentationRequest(StrictModel):
    """A caller presentation; admission canonicalizes and validates it."""

    components: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_NC_COMPONENTS,
    )
    strata: tuple[NormalCrossingsStratum, ...] = Field(
        min_length=1,
        max_length=MAX_NC_STRATA,
    )


class NormalCrossingsStratumRecord(StrictModel):
    """Per-stratum transport row bound to one admitted presentation."""

    components: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_NC_BRANCH_MULTIPLICITY,
    )
    dimension: int = Field(ge=0, le=MAX_NC_STRATUM_DIMENSION)
    branch_multiplicity: int = Field(ge=1, le=MAX_NC_BRANCH_MULTIPLICITY)

    @model_validator(mode="after")
    def require_bound_multiplicity(self) -> Self:
        if self.branch_multiplicity != len(self.components):
            raise _validation_error(
                "branch_multiplicity_binding",
                "branch multiplicity must equal the component-set cardinality",
            )
        return self


class StrataInCardinality(StrictModel):
    """The strata of one component-set cardinality in canonical order."""

    cardinality: int = Field(ge=1, le=MAX_NC_BRANCH_MULTIPLICITY)
    strata: tuple[tuple[VertexLabel, ...], ...] = Field(
        min_length=1,
        max_length=MAX_NC_STRATA,
    )

    @model_validator(mode="after")
    def require_canonical_group(self) -> Self:
        if any(len(stratum) != self.cardinality for stratum in self.strata):
            raise _validation_error(
                "cardinality_group_shape",
                "every stratum in a cardinality group must match its size",
            )
        if self.strata != tuple(sorted(set(self.strata))):
            raise _validation_error(
                "cardinality_group_order",
                "strata in a cardinality group must be sorted and distinct",
            )
        return self


class DualComplexResult(StrictModel):
    """The dual complex, stratum transport rows, and Cech incidence complex.

    ``dual_complex`` is the canonical finite simplicial complex with one
    vertex per component and one simplex per inclusion-maximal stratum
    component set; ``cech_value`` is the signed-incidence normalization /
    Cech complex with degree ``k`` spanned by the strata whose component
    sets have cardinality ``k + 1``.
    """

    presentation: NormalCrossingsPresentation
    strata: tuple[NormalCrossingsStratumRecord, ...] = Field(
        min_length=1,
        max_length=MAX_NC_STRATA,
    )
    dual_complex: FiniteSimplicialComplex
    strata_by_cardinality: tuple[StrataInCardinality, ...] = Field(
        min_length=1,
        max_length=MAX_NC_BRANCH_MULTIPLICITY,
    )
    cech_value: ChainComplexValue
    differential_squared_zero: tuple[BoundarySquareLedgerEntry, ...] = Field(
        default=(),
        max_length=MAX_NC_BRANCH_MULTIPLICITY,
    )

    @model_validator(mode="after")
    def require_structural_bindings(self) -> Self:
        presentation_keys = tuple(
            stratum.components for stratum in self.presentation.strata
        )
        if tuple(stratum.components for stratum in self.strata) != presentation_keys:
            raise _validation_error(
                "stratum_transport_binding",
                "stratum rows must replay the canonical presentation strata",
            )
        if any(
            record.dimension != stratum.dimension
            for record, stratum in zip(
                self.strata, self.presentation.strata, strict=True
            )
        ):
            raise _validation_error(
                "stratum_transport_binding",
                "stratum rows must retain the presented dimensions",
            )
        if self.dual_complex.vertices != self.presentation.components:
            raise _validation_error(
                "dual_complex_binding",
                "dual-complex vertices must be the presentation components",
            )
        cardinalities = tuple(group.cardinality for group in self.strata_by_cardinality)
        if cardinalities != tuple(range(1, len(cardinalities) + 1)):
            raise _validation_error(
                "cardinality_axis_invalid",
                "cardinality groups must cover contiguous sizes from one",
            )
        grouped = tuple(
            stratum for group in self.strata_by_cardinality for stratum in group.strata
        )
        if grouped != presentation_keys:
            raise _validation_error(
                "cardinality_partition_invalid",
                "cardinality groups must partition the presentation strata",
            )
        basis_sizes = tuple(len(group.strata) for group in self.strata_by_cardinality)
        if self.cech_value.basis_sizes != basis_sizes:
            raise _validation_error(
                "cech_basis_binding",
                "Cech basis sizes must match the cardinality strata counts",
            )
        if tuple(
            entry.upper_dimension for entry in self.differential_squared_zero
        ) != tuple(range(1, len(basis_sizes))):
            raise _validation_error(
                "square_ledger_incomplete",
                "the square-zero ledger must cover every adjacent degree pair",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class NearbyCycleLattice(StrictModel):
    """The integral nearby-cycle stalk lattice of one stratum.

    ``saturated_basis`` rows are the canonical basis vectors of
    ``K_S = ker(sum: Z^r -> Z)`` in sorted-branch ambient coordinates;
    ``milnor_fiber_cohomology_ranks[q]`` is the exact rank of
    ``H^q(MF_S; Z) = exterior^q Hom(K_S, Z)`` for ``q`` in ``0..r-1``.
    """

    components: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_NC_BRANCH_MULTIPLICITY,
    )
    branch_multiplicity: int = Field(ge=1, le=MAX_NC_BRANCH_MULTIPLICITY)
    lattice_rank: int = Field(ge=0, le=MAX_NC_BRANCH_MULTIPLICITY - 1)
    saturated_basis: IntegerMatrix
    milnor_fiber_cohomology_ranks: tuple[int, ...] = Field(
        min_length=1,
        max_length=MAX_NC_BRANCH_MULTIPLICITY,
    )
    basis_convention: Literal["SUCCESSIVE_DIFFERENCE_SORTED_BRANCHES"] = (
        "SUCCESSIVE_DIFFERENCE_SORTED_BRANCHES"
    )

    @model_validator(mode="after")
    def require_structural_lattice(self) -> Self:
        _require_sorted_distinct(self.components)
        if self.branch_multiplicity != len(self.components):
            raise _validation_error(
                "branch_multiplicity_binding",
                "branch multiplicity must equal the component-set cardinality",
            )
        if self.lattice_rank != self.branch_multiplicity - 1:
            raise _validation_error(
                "lattice_rank_binding",
                "the stalk lattice rank must be the branch multiplicity minus one",
            )
        if (
            self.saturated_basis.row_count != self.lattice_rank
            or self.saturated_basis.column_count != self.branch_multiplicity
        ):
            raise _validation_error(
                "lattice_basis_shape",
                "the saturated basis must be a (r-1) x r integer matrix",
            )
        if any(sum(row) != 0 for row in self.saturated_basis.entries) or any(
            abs(value) >= 10**MAX_NC_LATTICE_ENTRY_DIGITS
            for row in self.saturated_basis.entries
            for value in row
        ):
            raise _validation_error(
                "lattice_basis_kernel",
                "saturated basis vectors must lie in the sum-zero kernel",
            )
        if len(self.milnor_fiber_cohomology_ranks) != self.branch_multiplicity or any(
            rank < 0 for rank in self.milnor_fiber_cohomology_ranks
        ):
            raise _validation_error(
                "cohomology_rank_axis",
                "Milnor-fibre cohomology ranks must cover degrees 0..r-1",
            )
        return self


class SpecializationMap(StrictModel):
    """One stratum-inclusion specialization ``K_J -> K_I`` for ``I subset J``.

    ``source_components`` is the deeper stratum ``J`` and
    ``target_components`` the shallower stratum ``I``; the matrix is
    ``(rank K_J) x (rank K_I)`` in successive-difference bases.  Row ``k``
    expands the image of the ``k``-th source basis vector in the target
    basis, so lattice coordinates act as ``v -> v^T M`` (row convention)
    and compositions along ``H subset I subset J`` satisfy
    ``M_{H,J} = M_{I,J} M_{H,I}``.
    """

    source_components: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_NC_BRANCH_MULTIPLICITY,
    )
    target_components: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_NC_BRANCH_MULTIPLICITY,
    )
    matrix: IntegerMatrix
    fold_convention: Literal["FIBER_SUM_FOLD_TO_LAST_SORTED_BRANCH"] = (
        "FIBER_SUM_FOLD_TO_LAST_SORTED_BRANCH"
    )

    @model_validator(mode="after")
    def require_structural_specialization(self) -> Self:
        _require_sorted_distinct(self.source_components)
        _require_sorted_distinct(self.target_components)
        source = set(self.source_components)
        target = set(self.target_components)
        if not target < source:
            raise _validation_error(
                "specialization_inclusion_invalid",
                "the target component set must be a proper subset of the source",
            )
        if self.matrix.row_count != len(source) - 1 or self.matrix.column_count != (
            len(target) - 1
        ):
            raise _validation_error(
                "specialization_matrix_shape",
                "the matrix must be (rank K_J) x (rank K_I)",
            )
        return self


class NearbyCycleLatticesResult(StrictModel):
    """Per-stratum stalk lattices and cover specializations of a presentation."""

    presentation: NormalCrossingsPresentation
    lattices: tuple[NearbyCycleLattice, ...] = Field(
        min_length=1,
        max_length=MAX_NC_STRATA,
    )
    specializations: tuple[SpecializationMap, ...] = Field(
        default=(),
        max_length=MAX_NC_SPECIALIZATION_MAPS,
    )

    @model_validator(mode="after")
    def require_structural_bindings(self) -> Self:
        presentation_keys = tuple(
            stratum.components for stratum in self.presentation.strata
        )
        if tuple(lattice.components for lattice in self.lattices) != presentation_keys:
            raise _validation_error(
                "lattice_coverage",
                "stalk lattices must cover the canonical presentation strata once",
            )
        declared = set(presentation_keys)
        keys: list[tuple[tuple[VertexLabel, ...], tuple[VertexLabel, ...]]] = []
        for specialization in self.specializations:
            if (
                specialization.source_components not in declared
                or specialization.target_components not in declared
            ):
                raise _validation_error(
                    "specialization_unbound",
                    "specialization endpoints must be declared strata",
                )
            if len(specialization.source_components) != (
                len(specialization.target_components) + 1
            ):
                raise _validation_error(
                    "specialization_not_cover",
                    "published specialization maps cover exactly one added branch",
                )
            keys.append(
                (
                    specialization.target_components,
                    specialization.source_components,
                )
            )
        if keys != sorted(set(keys)):
            raise _validation_error(
                "specialization_order",
                "specializations must be distinct and sorted by (target, source)",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


def maximal_strata(
    keys: tuple[tuple[VertexLabel, ...], ...],
) -> tuple[tuple[VertexLabel, ...], ...]:
    """Return the inclusion-maximal component sets in canonical order."""

    maximal: list[tuple[VertexLabel, ...]] = []
    for key in keys:
        key_set = set(key)
        if not any(other != key and key_set < set(other) for other in keys):
            maximal.append(key)
    return tuple(sorted(maximal))


def stratum_subsets(
    key: tuple[VertexLabel, ...],
) -> tuple[tuple[VertexLabel, ...], ...]:
    """Return every nonempty subset of one component set in canonical order."""

    return tuple(
        sorted(
            subset
            for size in range(1, len(key) + 1)
            for subset in combinations(key, size)
        )
    )


__all__ = [
    "MAX_NC_BRANCH_MULTIPLICITY",
    "MAX_NC_COMPONENTS",
    "MAX_NC_LATTICE_ENTRY_DIGITS",
    "MAX_NC_SPECIALIZATION_CELLS",
    "MAX_NC_SPECIALIZATION_MAPS",
    "MAX_NC_STRATA",
    "MAX_NC_STRATUM_DIMENSION",
    "BoundarySquareLedgerEntry",
    "DualComplexResult",
    "NearbyCycleLattice",
    "NearbyCycleLatticesResult",
    "NormalCrossingsPresentation",
    "NormalCrossingsPresentationRequest",
    "NormalCrossingsStratum",
    "NormalCrossingsStratumRecord",
    "SpecializationMap",
    "StrataInCardinality",
    "maximal_strata",
    "stratum_subsets",
]
