"""Canonical finite holonomy extensions and exact torsion witnesses."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_INTEGER_DIGITS,
    CanonicalRational,
    DecimalIntegerEncoding,
    ExactInteger,
)
from jacobian._models import StrictModel
from jacobian.math.geometry.polytopes._models import (
    MAX_COMPUTED_FACETS,
    MAX_FACET_DIMENSION,
    CoordinateAxis,
    FacetIncidenceResult,
    RationalVPolytope,
)

MAX_EXTENSION_GROUP_ORDER = 8
MAX_EXTENSION_LATTICE_RANK = 4
MAX_ACTION_ENTRY_DIGITS = 1
MAX_COCYCLE_ENTRY_DIGITS = 4
MAX_EXTENSION_TORSION_RESULT_SIZE = 2_000_000
# The existing certified Smith carrier caps transformation entries at the
# canonical exact-integer limit. A reconstructed lattice solution multiplies a
# Smith right-transform entry by a transformed offset, so retain the resulting
# conservative digit envelope in the result type as well.
MAX_EXTENSION_PAIRING_DIGITS = MAX_CANONICAL_INTEGER_DIGITS + 16
MAX_EXTENSION_TORSION_VECTOR_DIGITS = 2 * MAX_CANONICAL_INTEGER_DIGITS + 16

GroupIndex = Annotated[StrictInt, Field(ge=0, le=MAX_EXTENSION_GROUP_ORDER - 1)]
ActionEntry = Annotated[int, DecimalIntegerEncoding(max_digits=MAX_ACTION_ENTRY_DIGITS)]
CocycleEntry = Annotated[
    int, DecimalIntegerEncoding(max_digits=MAX_COCYCLE_ENTRY_DIGITS)
]
GroupTableRow = Annotated[
    tuple[GroupIndex, ...],
    Field(min_length=1, max_length=MAX_EXTENSION_GROUP_ORDER),
]
ActionRow = Annotated[
    tuple[ActionEntry, ...],
    Field(min_length=1, max_length=MAX_EXTENSION_LATTICE_RANK),
]
ActionMatrix = Annotated[
    tuple[ActionRow, ...],
    Field(min_length=1, max_length=MAX_EXTENSION_LATTICE_RANK),
]
CocycleVector = Annotated[
    tuple[CocycleEntry, ...],
    Field(min_length=1, max_length=MAX_EXTENSION_LATTICE_RANK),
]
CocycleRow = Annotated[
    tuple[CocycleVector, ...],
    Field(min_length=1, max_length=MAX_EXTENSION_GROUP_ORDER),
]
ExtensionVector = Annotated[
    tuple[ExactInteger, ...],
    Field(min_length=1, max_length=MAX_EXTENSION_LATTICE_RANK),
]
ExtensionMatrixRow = Annotated[
    tuple[ExactInteger, ...],
    Field(min_length=1, max_length=MAX_EXTENSION_LATTICE_RANK),
]
ExtensionMatrix = Annotated[
    tuple[ExtensionMatrixRow, ...],
    Field(min_length=1, max_length=MAX_EXTENSION_LATTICE_RANK),
]
PairingInteger = Annotated[
    int, DecimalIntegerEncoding(max_digits=MAX_EXTENSION_PAIRING_DIGITS)
]
TorsionVectorInteger = Annotated[
    int, DecimalIntegerEncoding(max_digits=MAX_EXTENSION_TORSION_VECTOR_DIGITS)
]
TorsionVector = Annotated[
    tuple[TorsionVectorInteger, ...],
    Field(min_length=1, max_length=MAX_EXTENSION_LATTICE_RANK),
]
PairingTranslationInteger = Annotated[int, DecimalIntegerEncoding(max_digits=33)]
PairingTranslation = Annotated[
    tuple[PairingTranslationInteger, ...],
    Field(min_length=1, max_length=MAX_EXTENSION_LATTICE_RANK),
]


class FiniteLatticeExtension(StrictModel):
    """An extension ``0 -> Z^n -> Gamma -> G -> 0`` in a chosen section.

    ``multiplication_table`` labels the identity by 0. ``action_matrices[g]``
    is the left integral action rho(g) on column vectors. ``factor_set[g][h]``
    is the normalized integral 2-cocycle f(g,h). The associated group law is
    ``(v,g)(w,h)=(v+rho(g)w+f(g,h), gh)``. The operation checks all group,
    representation, faithfulness, normalization, and cocycle identities.

    The action matrices and factor-set entries have intentionally small initial
    envelopes; this keeps all exact verification and Smith inputs bounded.
    """

    multiplication_table: Annotated[
        tuple[GroupTableRow, ...],
        Field(min_length=1, max_length=MAX_EXTENSION_GROUP_ORDER),
    ]
    action_matrices: Annotated[
        tuple[ActionMatrix, ...],
        Field(min_length=1, max_length=MAX_EXTENSION_GROUP_ORDER),
    ]
    factor_set: Annotated[
        tuple[CocycleRow, ...],
        Field(min_length=1, max_length=MAX_EXTENSION_GROUP_ORDER),
    ]

    @model_validator(mode="after")
    def require_coherent_shapes(self) -> Self:
        order = len(self.multiplication_table)
        rank = len(self.action_matrices[0]) if self.action_matrices else 0
        if order > MAX_EXTENSION_GROUP_ORDER:
            raise _error("shape", "finite holonomy exceeds its admitted order")
        if not 1 <= rank <= MAX_EXTENSION_LATTICE_RANK:
            raise _error("shape", "lattice rank is outside the admitted envelope")
        if len(self.action_matrices) != order or len(self.factor_set) != order:
            raise _error("shape", "table, action, and factor-set orders must agree")
        if any(len(row) != order for row in self.multiplication_table):
            raise _error("shape", "multiplication table must be square")
        if any(
            len(matrix) != rank or any(len(row) != rank for row in matrix)
            for matrix in self.action_matrices
        ):
            raise _error("shape", "every action matrix must match the lattice rank")
        if any(
            len(row) != order or any(len(vector) != rank for vector in row)
            for row in self.factor_set
        ):
            raise _error("shape", "factor set must have shape |G| by |G| by rank")
        return self


class NonTorsionLiftObstruction(StrictModel):
    """Smith obstruction proving no lift of one nonidentity g has finite order."""

    holonomy_element: GroupIndex
    holonomy_order: StrictInt = Field(ge=2, le=MAX_EXTENSION_GROUP_ORDER)
    norm_matrix: ExtensionMatrix
    power_offset: ExtensionVector
    obstruction_vector: ExtensionVector
    modulus: Annotated[ExactInteger, Field(ge=0)]
    pairing: PairingInteger

    @model_validator(mode="after")
    def require_obstruction_shape(self) -> Self:
        rank = len(self.norm_matrix)
        if (
            any(len(row) != rank for row in self.norm_matrix)
            or len(self.power_offset) != rank
            or len(self.obstruction_vector) != rank
        ):
            raise _error(
                "certificate_shape",
                "lift obstruction vectors must match the norm-matrix rank",
            )
        return self


class TorsionLiftWitness(StrictModel):
    """A concrete finite-order nonidentity extension element ``(v,g)``."""

    holonomy_element: GroupIndex
    holonomy_order: StrictInt = Field(ge=2, le=MAX_EXTENSION_GROUP_ORDER)
    translation_part: TorsionVector
    norm_matrix: ExtensionMatrix
    power_offset: ExtensionVector

    @model_validator(mode="after")
    def require_square_norm_shape(self) -> Self:
        rank = len(self.norm_matrix)
        if (
            any(len(row) != rank for row in self.norm_matrix)
            or len(self.translation_part) != rank
            or len(self.power_offset) != rank
        ):
            raise _error(
                "witness_shape",
                "torsion witness coordinates must match its norm matrix",
            )
        return self


class CrystallographicExtensionTorsionResult(StrictModel):
    """Exact torsion-freeness result retaining the full extension source."""

    source: FiniteLatticeExtension
    torsion_free: bool
    torsion_witness: TorsionLiftWitness | None
    lift_obstructions: tuple[NonTorsionLiftObstruction, ...] = Field(
        max_length=MAX_EXTENSION_GROUP_ORDER - 1
    )
    conclusion: Literal["TORSION_FREE", "HAS_TORSION"]

    @model_validator(mode="after")
    def require_aligned_conclusion(self) -> Self:
        if self.torsion_free != (self.conclusion == "TORSION_FREE"):
            raise _error("result", "torsion-free flag and conclusion disagree")
        if self.torsion_free:
            if self.torsion_witness is not None:
                raise _error(
                    "result", "torsion-free result cannot include a torsion witness"
                )
            expected = len(self.source.multiplication_table) - 1
            if len(self.lift_obstructions) != expected:
                raise _error(
                    "result",
                    "one non-torsion obstruction is required per nonidentity element",
                )
            expected_elements = tuple(range(1, len(self.source.multiplication_table)))
            actual_elements = tuple(
                item.holonomy_element for item in self.lift_obstructions
            )
            if actual_elements != expected_elements:
                raise _error(
                    "result",
                    "lift obstructions must cover nonidentity elements in table order",
                )
            if any(
                item.holonomy_element >= len(self.source.multiplication_table)
                for item in self.lift_obstructions
            ):
                raise _error(
                    "result", "lift obstruction element is outside the source table"
                )
            rank = len(self.source.action_matrices[0])
            if any(
                len(item.norm_matrix) != rank
                or any(len(row) != rank for row in item.norm_matrix)
                or len(item.power_offset) != rank
                for item in self.lift_obstructions
            ):
                raise _error(
                    "result",
                    "lift obstruction dimensions differ from the source lattice",
                )
        elif self.torsion_witness is None or self.lift_obstructions:
            raise _error(
                "result",
                "torsion result requires one witness and no negative certificates",
            )
        elif (
            self.torsion_witness.holonomy_element == 0
            or self.torsion_witness.holonomy_element
            >= len(self.source.multiplication_table)
        ):
            raise _error(
                "result", "torsion witness must project to a nonidentity source element"
            )
        elif len(self.torsion_witness.norm_matrix) != len(
            self.source.action_matrices[0]
        ):
            raise _error(
                "result", "torsion witness dimensions differ from the source lattice"
            )
        return self


class CrystallographicAffineSectionMap(StrictModel):
    """One affine map for a chosen finite-holonomy section representative.

    Its translation is the rational section shift ``q_g``. For a nonzero
    extension cocycle these maps do not form a representation of the finite
    holonomy group: composing two section maps introduces the lattice
    translation ``f(g,h)``.
    """

    holonomy_element: GroupIndex
    linear_part: ActionMatrix
    section_shift: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_EXTENSION_LATTICE_RANK
    )

    @model_validator(mode="after")
    def require_square_map_shape(self) -> Self:
        rank = len(self.linear_part)
        if (
            not 1 <= rank <= MAX_EXTENSION_LATTICE_RANK
            or any(len(row) != rank for row in self.linear_part)
            or len(self.section_shift) != rank
        ):
            raise _error(
                "affine_map_shape",
                "affine section map must be square and match the lattice rank",
            )
        return self


class CrystallographicAffineRealization(StrictModel):
    """The exact affine realization of the chosen section of an extension.

    The retained maps are ``A_g(x)=rho(g)x+q_g``. The full extension acts by
    ``(v,g) -> T_v o A_g``, with ``T_v(x)=x+v``; this realizes the extension's
    multiplication law, while the section maps alone compose with the cocycle
    translation.
    """

    source: FiniteLatticeExtension
    section_maps: tuple[CrystallographicAffineSectionMap, ...] = Field(
        min_length=1, max_length=MAX_EXTENSION_GROUP_ORDER
    )

    @model_validator(mode="after")
    def require_source_bound_maps(self) -> Self:
        if len(self.section_maps) != len(self.source.multiplication_table):
            raise _error(
                "affine_map_order",
                "there must be one affine section map per holonomy element",
            )
        rank = len(self.source.action_matrices[0])
        for index, (section_map, action) in enumerate(
            zip(self.section_maps, self.source.action_matrices, strict=True)
        ):
            if (
                section_map.holonomy_element != index
                or section_map.linear_part != action
                or len(section_map.section_shift) != rank
            ):
                raise _error(
                    "affine_map_source",
                    "affine section maps must retain source order, action, and rank",
                )
        return self


class PolytopeFacetPairing(StrictModel):
    """One directed side pairing by an element ``(v,g)`` of the extension."""

    source_facet_index: StrictInt = Field(ge=0, le=MAX_COMPUTED_FACETS - 1)
    target_facet_index: StrictInt = Field(ge=0, le=MAX_COMPUTED_FACETS - 1)
    lattice_translation: PairingTranslation
    holonomy_element: GroupIndex


class CrystallographicPolytopePairingRequest(StrictModel):
    """A bounded rational polytope with a complete proposed facet ledger."""

    affine_realization: CrystallographicAffineRealization
    polytope: RationalVPolytope
    lattice_axes: tuple[CoordinateAxis, ...] = Field(
        min_length=1, max_length=MAX_FACET_DIMENSION
    )
    pairings: tuple[PolytopeFacetPairing, ...] = Field(
        min_length=2, max_length=MAX_COMPUTED_FACETS
    )


class CrystallographicPolytopePairing(StrictModel):
    """Canonical exact facet pairing data bound to its source polytope."""

    source_facet_index: StrictInt = Field(ge=0, le=MAX_COMPUTED_FACETS - 1)
    target_facet_index: StrictInt = Field(ge=0, le=MAX_COMPUTED_FACETS - 1)
    lattice_translation: PairingTranslation
    holonomy_element: GroupIndex


class CrystallographicPolytopePairingResult(StrictModel):
    """Source-bound complete facet profile and validated directed pairings."""

    affine_realization: CrystallographicAffineRealization
    polytope: RationalVPolytope
    lattice_axes: tuple[CoordinateAxis, ...]
    facet_profile: FacetIncidenceResult
    pairings: tuple[CrystallographicPolytopePairing, ...] = Field(
        min_length=2, max_length=MAX_COMPUTED_FACETS
    )

    @model_validator(mode="after")
    def require_retained_sources_align(self) -> Self:
        rank = len(self.affine_realization.source.action_matrices[0])
        if (
            self.lattice_axes != self.polytope.space.axes
            or len(self.lattice_axes) != rank
            or self.facet_profile.dimension != rank
            or len(self.facet_profile.vertices) != len(self.polytope.vertices)
            or tuple(vertex.coordinates for vertex in self.facet_profile.vertices)
            != tuple(vertex.coordinates for vertex in self.polytope.vertices)
            or tuple(item.source_facet_index for item in self.pairings)
            != tuple(range(len(self.facet_profile.facets)))
            or len(self.pairings) != len(self.facet_profile.facets)
        ):
            raise _error(
                "polytope_pairing_result",
                "result axes, facet profile, pairings, or retained sources do not align",
            )
        return self


class CrystallographicFundamentalDomainResult(StrictModel):
    """Exact bounded overlap and covolume check for a paired polytope."""

    source: CrystallographicPolytopePairingResult
    is_fundamental_domain: bool
    polytope_volume: CanonicalRational
    quotient_covolume: CanonicalRational
    overlap_translation: PairingTranslation | None
    overlap_holonomy_element: GroupIndex | None

    @model_validator(mode="after")
    def require_aligned_witness(self) -> Self:
        if (self.overlap_translation is None) != (
            self.overlap_holonomy_element is None
        ):
            raise _error(
                "fundamental_domain_result",
                "overlap witness fields must occur together",
            )
        if self.is_fundamental_domain and (
            self.overlap_translation is not None
            or self.polytope_volume != self.quotient_covolume
        ):
            raise _error(
                "fundamental_domain_result",
                "positive result requires equal covolume and no overlap",
            )
        if not self.is_fundamental_domain and (
            self.polytope_volume == self.quotient_covolume
            and self.overlap_translation is None
        ):
            raise _error(
                "fundamental_domain_result",
                "negative result requires unequal covolume or an overlap witness",
            )
        return self


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"crystallographic.extension.{reason}", message)


__all__ = [
    "MAX_ACTION_ENTRY_DIGITS",
    "MAX_COCYCLE_ENTRY_DIGITS",
    "MAX_EXTENSION_GROUP_ORDER",
    "MAX_EXTENSION_LATTICE_RANK",
    "MAX_EXTENSION_PAIRING_DIGITS",
    "MAX_EXTENSION_TORSION_RESULT_SIZE",
    "MAX_EXTENSION_TORSION_VECTOR_DIGITS",
    "CrystallographicAffineRealization",
    "CrystallographicAffineSectionMap",
    "CrystallographicExtensionTorsionResult",
    "CrystallographicFundamentalDomainResult",
    "FiniteLatticeExtension",
    "NonTorsionLiftObstruction",
    "TorsionLiftWitness",
]
