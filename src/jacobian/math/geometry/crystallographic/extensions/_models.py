"""Canonical finite holonomy extensions and exact torsion witnesses."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Any, Literal, Self, cast

from pydantic import (
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    GetPydanticSchema,
    StrictInt,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError, core_schema

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
    MAX_FACET_INCIDENCES,
    MAX_VERTICES,
    CoordinateAxis,
    FacetIncidenceResult,
    PrimitiveFacet,
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.values import Halfspace, Vertex
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
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
FaceOrbitGroupTranslationEntry = Annotated[
    int,
    DecimalIntegerEncoding(max_digits=128),
]
FaceOrbitGroupTranslation = Annotated[
    tuple[FaceOrbitGroupTranslationEntry, ...],
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
            or any(
                item.target_facet_index >= len(self.facet_profile.facets)
                or item.holonomy_element >= len(self.affine_realization.section_maps)
                or len(item.lattice_translation) != rank
                for item in self.pairings
            )
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


_NATIVE_SOURCE_MODELS = (
    CrystallographicFundamentalDomainResult,
    CrystallographicPolytopePairingResult,
    CrystallographicAffineRealization,
    CrystallographicAffineSectionMap,
    FiniteLatticeExtension,
    CrystallographicPolytopePairing,
    RationalVPolytope,
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    FacetIncidenceResult,
    PrimitiveFacet,
    Halfspace,
    Vertex,
    CanonicalRational,
    ChainComplexValue,
)


def _bounded_native_source(value: object) -> object:
    """Revalidate bypassed native carriers without invoking their serializers.

    A source has at most MAX_FACET_INCIDENCES incidence leaves, plus bounded
    vertex/facet records. Sixteen nodes per possible incidence, facet, or
    vertex conservatively covers every declared container and rational leaf;
    the source schema has fewer than sixteen nested container levels. These
    are structural preflight bounds, not another geometric admission or solve.
    """
    remaining = 16 * (MAX_FACET_INCIDENCES + MAX_COMPUTED_FACETS + MAX_VERTICES)
    max_fields = max(len(model.model_fields) for model in _NATIVE_SOURCE_MODELS)

    def project(item: object, depth: int) -> object:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 16:
            raise _error(
                "source_bound", "native source exceeds its structural envelope"
            )
        if type(item) is CoefficientRing:
            return item
        if item is None or type(item) is bool:
            return item
        if type(item) is int:
            if item.bit_length() > 4 * MAX_CANONICAL_INTEGER_DIGITS:
                raise _error(
                    "source_bound",
                    "native source scalar exceeds its exact encoding envelope",
                )
            return item
        if type(item) is str:
            if len(item) > MAX_CANONICAL_INTEGER_DIGITS:
                raise _error(
                    "source_bound", "native source text exceeds its encoding envelope"
                )
            return item
        if type(item) in (
            *_NATIVE_SOURCE_MODELS,
            BieberbachFaceOrbitMap,
            BieberbachGroupRingBoundaryEntry,
        ):
            model = cast(StrictModel, item)
            fields = type(model).model_fields
            contents = vars(model)
            if len(contents) != len(fields) or any(
                key not in fields for key in contents
            ):
                raise _error(
                    "source_shape",
                    "native source model fields are incomplete or malformed",
                )
            return {key: project(contents[key], depth + 1) for key in fields}
        if type(item) is dict:
            if len(item) > max_fields or any(type(key) is not str for key in item):
                raise _error(
                    "source_shape", "native source record fields are malformed"
                )
            return {key: project(child, depth + 1) for key, child in item.items()}
        if type(item) is tuple or type(item) is list:
            if len(item) > MAX_FACET_INCIDENCES:
                raise _error(
                    "source_bound",
                    "native source container exceeds its structural envelope",
                )
            return tuple(project(child, depth + 1) for child in item)
        raise _error(
            "source_shape",
            "native source must contain canonical values and ordinary containers",
        )

    return project(value, 0)


def _native_source_schema(
    source_type: Any, handler: GetCoreSchemaHandler
) -> core_schema.CoreSchema:
    schema = handler(source_type)
    return core_schema.json_or_python_schema(
        json_schema=schema,
        python_schema=core_schema.no_info_before_validator_function(
            _bounded_native_source, schema
        ),
    )


_BoundedFundamentalDomainSource = Annotated[
    CrystallographicFundamentalDomainResult, GetPydanticSchema(_native_source_schema)
]


_BoundedChainComplex = Annotated[
    ChainComplexValue, GetPydanticSchema(_native_source_schema)
]


class BieberbachFaceOrbitMap(StrictModel):
    """One exact vertex map induced by a directed paired polygon edge."""

    source_facet_index: StrictInt = Field(ge=0, le=MAX_COMPUTED_FACETS - 1)
    source_vertex_index: StrictInt = Field(ge=0, le=MAX_VERTICES - 1)
    target_facet_index: StrictInt = Field(ge=0, le=MAX_COMPUTED_FACETS - 1)
    target_vertex_index: StrictInt = Field(ge=0, le=MAX_VERTICES - 1)
    lattice_translation: PairingTranslation
    holonomy_element: GroupIndex


class BieberbachGroupRingBoundaryEntry(StrictModel):
    """A signed face incidence labelled by an element of the extension."""

    source_cell_index: StrictInt = Field(ge=0, le=MAX_VERTICES - 1)
    target_cell_index: StrictInt = Field(ge=0, le=MAX_VERTICES - 1)
    coefficient: StrictInt = Field(ge=-1, le=1)
    incidence_index: StrictInt = Field(ge=0, le=MAX_VERTICES - 1)
    lattice_translation: FaceOrbitGroupTranslation
    holonomy_element: GroupIndex


MAX_FACE_ORBIT_VERTICES = 32
_FACE_ORBIT_COLLECTION_LIMITS = {
    "vertex_orbits": MAX_COMPUTED_FACETS,
    "edge_orbit_representatives": MAX_COMPUTED_FACETS,
    "orbit_maps": MAX_COMPUTED_FACETS * MAX_FACE_ORBIT_VERTICES,
    "boundary_1_to_0": 2 * MAX_COMPUTED_FACETS * MAX_FACE_ORBIT_VERTICES,
    "boundary_2_to_1": MAX_COMPUTED_FACETS,
}


def _face_orbit_json_entries(
    entries: list[object] | tuple[object, ...],
) -> tuple[object, ...]:
    """Project only the declared translation tuple after admitting each row."""
    projected: list[object] = []
    for entry in entries:
        if type(entry) is not dict:
            raise _error(
                "face_orbit_collection", "face-orbit ledger rows must be objects"
            )
        # Both endpoint maps and group-ring incidences declare six fields.
        if len(entry) > 6:
            raise _error(
                "face_orbit_bound", "face-orbit ledger row exceeds its field bound"
            )
        translation = entry.get("lattice_translation")
        if type(translation) is list:
            if len(translation) > MAX_EXTENSION_LATTICE_RANK:
                raise _error(
                    "face_orbit_bound", "face-orbit translation exceeds its rank bound"
                )
            entry = {**entry, "lattice_translation": tuple(translation)}
        projected.append(entry)
    return tuple(projected)


def _affine_endpoint_matches(
    realization: CrystallographicAffineRealization,
    item: BieberbachFaceOrbitMap,
    source_coordinate: tuple[CanonicalRational, ...],
    target_coordinates: tuple[tuple[CanonicalRational, ...], ...],
    target_vertex_index: int,
) -> bool:
    """Return whether an orbit map really lands on its declared endpoint.

    The pairing acts affinely as ``A_g(x) = linear_part x + section_shift +
    translation``. Checking only that the declared target vertex is *some*
    vertex of the target facet leaves a payload free to swap it for the other
    endpoint of the same edge, which is a mathematically false orbit map.
    """
    section_map = realization.section_maps[item.holonomy_element]
    dimension = len(source_coordinate)
    if len(target_coordinates[target_vertex_index]) != dimension:
        return False
    for row in range(dimension):
        image = section_map.section_shift[row].as_fraction() + Fraction(
            item.lattice_translation[row]
        )
        for column in range(dimension):
            image += section_map.linear_part[row][column] * (
                source_coordinate[column].as_fraction()
            )
        if image != target_coordinates[target_vertex_index][row].as_fraction():
            return False
    return True


class BieberbachFaceOrbitComplex(StrictModel):
    """Two-dimensional quotient cell structure and its integral chains.

    The source retains the checked fundamental polygon and all directed side
    pairings. Orbit maps record every paired-edge endpoint identification.
    Group-ring boundary entries preserve the exact deck transformations. The
    ordinary ZZ chain complex is the augmentation of these cellular chains;
    this value does not claim to contain or verify a free ZGamma-resolution.
    """

    model_config = ConfigDict(revalidate_instances="always")

    source: _BoundedFundamentalDomainSource
    # Schema bounds describe the envelope; the raw field preflight below also
    # rejects oversized malformed rows before nested error expansion.
    vertex_orbits: tuple[
        Annotated[tuple[StrictInt, ...], Field(max_length=MAX_FACE_ORBIT_VERTICES)],
        ...,
    ] = Field(max_length=MAX_COMPUTED_FACETS)
    edge_orbit_representatives: tuple[StrictInt, ...] = Field(
        max_length=MAX_COMPUTED_FACETS
    )
    orbit_maps: tuple[BieberbachFaceOrbitMap, ...] = Field(
        max_length=MAX_COMPUTED_FACETS * MAX_FACE_ORBIT_VERTICES
    )
    boundary_1_to_0: tuple[BieberbachGroupRingBoundaryEntry, ...] = Field(
        max_length=2 * MAX_COMPUTED_FACETS * MAX_FACE_ORBIT_VERTICES
    )
    boundary_2_to_1: tuple[BieberbachGroupRingBoundaryEntry, ...] = Field(
        max_length=MAX_COMPUTED_FACETS
    )
    quotient_chain_complex: _BoundedChainComplex

    @field_validator(
        "vertex_orbits",
        "edge_orbit_representatives",
        "orbit_maps",
        "boundary_1_to_0",
        "boundary_2_to_1",
        mode="before",
    )
    @classmethod
    def require_bounded_raw_collections(
        cls, value: object, info: ValidationInfo
    ) -> object:
        if type(value) is not list and type(value) is not tuple:
            raise _error(
                "face_orbit_collection",
                "face-orbit collections must be ordinary tuples or arrays",
            )
        field = info.field_name
        if field not in _FACE_ORBIT_COLLECTION_LIMITS:
            raise _error("face_orbit_collection", "unknown face-orbit collection field")
        if len(value) > _FACE_ORBIT_COLLECTION_LIMITS[field]:
            raise _error(
                "face_orbit_bound",
                "face-orbit collection exceeds its admitted length",
            )
        if field == "vertex_orbits":
            for orbit in value:
                if type(orbit) is not list and type(orbit) is not tuple:
                    raise _error(
                        "face_orbit_collection",
                        "vertex-orbit rows must be ordinary tuples or arrays",
                    )
                if len(orbit) > MAX_FACE_ORBIT_VERTICES:
                    raise _error(
                        "face_orbit_bound",
                        "vertex-orbit row exceeds its admitted length",
                    )
        # Strict JSON loses its array provenance after a before-field hook.
        # Project only declared tuple paths after admission; scalar containers
        # remain untouched for their leaf validators to reject without copying.
        if info.mode != "json":
            return _bounded_native_source(value)
        if field == "vertex_orbits":
            return tuple(tuple(orbit) for orbit in value)
        if field == "edge_orbit_representatives":
            return tuple(value)
        return _face_orbit_json_entries(value)

    @model_validator(mode="after")
    def require_bounded_two_dimensional_source(self) -> Self:
        profile = self.source.source.facet_profile
        source_vertices = set(range(len(profile.vertices)))
        expected_maps = {
            (facet_index, vertex_index)
            for facet_index, facet in enumerate(profile.facets)
            for vertex_index in facet.source_vertex_indices
        }
        actual_maps = {
            (item.source_facet_index, item.source_vertex_index)
            for item in self.orbit_maps
        }
        pairing_by_source = {
            pairing.source_facet_index: pairing
            for pairing in self.source.source.pairings
        }
        maps_match_facets = all(
            item.source_facet_index in pairing_by_source
            and item.source_vertex_index
            in profile.facets[item.source_facet_index].source_vertex_indices
            and item.target_facet_index < len(profile.facets)
            and item.target_facet_index
            == pairing_by_source[item.source_facet_index].target_facet_index
            and item.target_vertex_index
            in profile.facets[item.target_facet_index].source_vertex_indices
            and item.lattice_translation
            == pairing_by_source[item.source_facet_index].lattice_translation
            and item.holonomy_element
            == pairing_by_source[item.source_facet_index].holonomy_element
            for item in self.orbit_maps
        )
        flattened_orbits = tuple(
            vertex for orbit in self.vertex_orbits for vertex in orbit
        )
        canonical_vertex_orbits = tuple(
            tuple(sorted(orbit))
            for orbit in sorted(
                self.vertex_orbits, key=lambda orbit: min(orbit, default=-1)
            )
            if orbit
        )
        expected_edge_representatives = {
            min(pairing.source_facet_index, pairing.target_facet_index)
            for pairing in self.source.source.pairings
        }
        if (
            not self.source.is_fundamental_domain
            or profile.dimension != 2
            or self.quotient_chain_complex.coefficient_ring != CoefficientRing.INTEGER
            or self.quotient_chain_complex.degree_min != 0
            or self.quotient_chain_complex.degree_max != 2
            or actual_maps != expected_maps
            or len(actual_maps) != len(self.orbit_maps)
            or not maps_match_facets
            or set(flattened_orbits) != source_vertices
            or len(flattened_orbits) != len(source_vertices)
            or self.vertex_orbits != canonical_vertex_orbits
            or set(self.edge_orbit_representatives) != expected_edge_representatives
            or self.edge_orbit_representatives
            != tuple(sorted(expected_edge_representatives))
            or len(self.edge_orbit_representatives)
            != len(expected_edge_representatives)
            or self.quotient_chain_complex.basis_sizes
            != (len(self.vertex_orbits), len(self.edge_orbit_representatives), 1)
            or len(self.boundary_1_to_0) != 2 * len(self.edge_orbit_representatives)
            or len(self.boundary_2_to_1) != len(profile.facets)
            or {
                (entry.source_cell_index, entry.incidence_index)
                for entry in self.boundary_1_to_0
            }
            != {
                (edge_index, vertex_index)
                for edge_index, facet_index in enumerate(
                    self.edge_orbit_representatives
                )
                for vertex_index in profile.facets[facet_index].source_vertex_indices
            }
            or {entry.incidence_index for entry in self.boundary_2_to_1}
            != set(range(len(profile.facets)))
        ):
            raise _error(
                "face_orbit_source",
                "face-orbit source, complete endpoint maps, orbit partition, edge representatives, or augmented ZZ chain axes are inconsistent",
            )
        if any(
            entry.source_cell_index >= len(self.edge_orbit_representatives)
            or entry.target_cell_index >= len(self.vertex_orbits)
            for entry in self.boundary_1_to_0
        ) or any(
            entry.source_cell_index != 0
            or entry.target_cell_index >= len(self.edge_orbit_representatives)
            for entry in self.boundary_2_to_1
        ):
            raise _error(
                "face_orbit_boundary_axis",
                "group-labelled boundary entry has an invalid cell index",
            )
        # The retained differentials must be exactly the augmentations of the
        # retained group-ring incidences. Checking only the basis axes lets a
        # payload swap in a different but still correctly shaped complex, which
        # would then report homology the retained incidences do not support.
        # d1 has one row per vertex orbit and one column per edge orbit; d2 has
        # one row per edge orbit and the single top cell as its only column.
        # Every orbit map must actually land on the endpoint the retained
        # pairing sends its source vertex to, not merely on some vertex of the
        # target facet. The affine image is evaluated once per map, which is
        # bounded by the facet and vertex envelopes rather than a search.
        realization = self.source.source.affine_realization
        coordinates = tuple(tuple(vertex.coordinates) for vertex in profile.vertices)
        if not all(
            _affine_endpoint_matches(
                realization,
                item,
                coordinates[item.source_vertex_index],
                coordinates,
                item.target_vertex_index,
            )
            for item in self.orbit_maps
        ):
            raise _error(
                "face_orbit_endpoint",
                "orbit map target vertex is not the affine image of its source vertex",
            )
        edge_count = len(self.edge_orbit_representatives)
        vertex_count = len(self.vertex_orbits)
        signed_1: dict[tuple[int, int], int] = {}
        for entry in self.boundary_1_to_0:
            key = (entry.target_cell_index, entry.source_cell_index)
            signed_1[key] = signed_1.get(key, 0) + entry.coefficient
        signed_2: dict[int, int] = {}
        for entry in self.boundary_2_to_1:
            signed_2[entry.target_cell_index] = (
                signed_2.get(entry.target_cell_index, 0) + entry.coefficient
            )
        expected_d1 = tuple(
            tuple(
                signed_1.get((vertex_index, edge_index), 0)
                for edge_index in range(edge_count)
            )
            for vertex_index in range(vertex_count)
        )
        expected_d2 = tuple(
            (signed_2.get(edge_index, 0),) for edge_index in range(edge_count)
        )
        if self.quotient_chain_complex.differential_matrices != (
            expected_d1,
            expected_d2,
        ):
            raise _error(
                "face_orbit_differential",
                "quotient differentials must be the augmentations of the retained boundary incidences",
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
    "BieberbachFaceOrbitComplex",
    "BieberbachFaceOrbitMap",
    "BieberbachGroupRingBoundaryEntry",
    "CrystallographicAffineRealization",
    "CrystallographicAffineSectionMap",
    "CrystallographicExtensionTorsionResult",
    "CrystallographicFundamentalDomainResult",
    "FiniteLatticeExtension",
    "NonTorsionLiftObstruction",
    "TorsionLiftWitness",
]
