"""Typed wire contracts for exact rational toric-geometry operations."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.matrices.values import IntegerMatrix

if TYPE_CHECKING:
    from jacobian.math.geometry.toric._kernel import RecognizedFan

MAX_TORIC_LATTICE_RANK = 4
MAX_TORIC_RAY_COUNT = 12
MAX_TORIC_CONE_COUNT = 40
MAX_TORIC_CONE_GENERATORS = 8
MAX_TORIC_COORDINATE_DIGITS = 8

# Affine-chart envelope: the double-description dual cone, the lattice points of
# the fundamental parallelepiped, the Hilbert basis, and the relation lattice
# are each bounded before any exact enumeration starts. A cone whose derived
# quantities leave the envelope is refused, never silently truncated.
MAX_TORIC_CHART_BOX = 20_000
MAX_TORIC_CHART_CANDIDATES = 1_024
MAX_TORIC_CHART_DUAL_RAYS = 32
MAX_TORIC_CHART_GENERATORS = 96
MAX_TORIC_CHART_RELATIONS = 192
MAX_TORIC_CHART_COMPONENT_DIGITS = 512

# Toric-morphism envelope: exact cone-membership checks summed over every
# source-ray/target-cone pair.
MAX_TORIC_MORPHISM_WORK = 200_000

ToricRayIndex = StrictInt

_ENVELOPE_DESCRIPTION = (
    f"lattice rank at most {MAX_TORIC_LATTICE_RANK}, at most "
    f"{MAX_TORIC_RAY_COUNT} rays, at most {MAX_TORIC_CONE_COUNT} cones, at most "
    f"{MAX_TORIC_CONE_GENERATORS} generators per cone, and at most "
    f"{MAX_TORIC_COORDINATE_DIGITS} decimal digits per coordinate"
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"toric.{reason}", message)


class ToricFanPresentation(StrictModel):
    """A proposed finite rational fan in the standard lattice ``Z^n``.

    Structural wire shape only: ray and cone index tuples are bounded and
    canonically ordered. Mathematical recognition (primitivity, strong
    convexity, extreme rays, face closure, common-face intersections) belongs
    to the owner's exact kernel, never to this model.
    """

    lattice_rank: StrictInt = Field(
        ge=1,
        le=MAX_TORIC_LATTICE_RANK,
        description=f"Ambient lattice rank n; at most {MAX_TORIC_LATTICE_RANK}.",
    )
    rays: tuple[tuple[ExactInteger, ...], ...] = Field(
        max_length=MAX_TORIC_RAY_COUNT,
        description=(
            f"Primitive integer ray generators v_rho in Z^n; at most "
            f"{MAX_TORIC_RAY_COUNT} rays with at most "
            f"{MAX_TORIC_COORDINATE_DIGITS} decimal digits per coordinate."
        ),
    )
    cones: tuple[tuple[ToricRayIndex, ...], ...] = Field(
        max_length=MAX_TORIC_CONE_COUNT,
        description=(
            f"Cones as strictly increasing unique ray-index tuples; the empty "
            f"tuple is the zero cone. At most {MAX_TORIC_CONE_COUNT} cones with "
            f"at most {MAX_TORIC_CONE_GENERATORS} generators each."
        ),
    )

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:
        limit = 10**MAX_TORIC_COORDINATE_DIGITS
        for ray in self.rays:
            if len(ray) != self.lattice_rank:
                raise _validation_error(
                    "ray_length_matches_lattice_rank",
                    "every ray must have exactly lattice_rank coordinates",
                )
            if any(abs(value) >= limit for value in ray):
                raise _validation_error(
                    "coordinates_exceed_envelope",
                    "ray coordinates exceed the "
                    f"{MAX_TORIC_COORDINATE_DIGITS}-digit toric envelope",
                )
        seen_cones: set[tuple[int, ...]] = set()
        for cone in self.cones:
            if len(cone) > MAX_TORIC_CONE_GENERATORS:
                raise _validation_error(
                    "cone_generators_exceed_envelope",
                    "a cone exceeds the "
                    f"{MAX_TORIC_CONE_GENERATORS}-generator toric envelope",
                )
            if any(index < 0 or index >= len(self.rays) for index in cone):
                raise _validation_error(
                    "cone_ray_index_out_of_range",
                    "cone ray indices must address declared rays",
                )
            if any(first >= second for first, second in itertools.pairwise(cone)):
                raise _validation_error(
                    "cone_ray_indices_strictly_increasing",
                    "cone ray indices must be strictly increasing",
                )
            if cone in seen_cones:
                raise _validation_error(
                    "duplicate_cone", "cones must be declared at most once"
                )
            seen_cones.add(cone)
        return self


class CharacterVector(StrictModel):
    """One exact lattice character ``m`` of the dual lattice ``M = Z^n``."""

    entries: tuple[ExactInteger, ...] = Field(
        min_length=1,
        max_length=MAX_TORIC_LATTICE_RANK,
        description=(
            f"Integer character coordinates with at most "
            f"{MAX_TORIC_COORDINATE_DIGITS} decimal digits each."
        ),
    )


class FanValidationRequest(StrictModel):
    """Recognize one proposed finite rational fan presentation."""

    fan: ToricFanPresentation = Field(
        description=f"Proposed fan presentation; envelope: {_ENVELOPE_DESCRIPTION}."
    )


class FanValidationResult(StrictModel):
    """Source-bound recognition verdict for one proposed fan."""

    fan: ToricFanPresentation = Field(description="The retained source fan.")
    status: Literal["VALID", "INVALID"]
    obstruction_code: str | None = Field(
        default=None,
        description=(
            "Stable owner code of the first exact obstruction; present only "
            "when the fan is INVALID."
        ),
    )
    obstruction_message: str | None = Field(default=None)

    @model_validator(mode="after")
    def require_bound_obstruction(self) -> Self:
        if self.status == "VALID":
            if (
                self.obstruction_code is not None
                or self.obstruction_message is not None
            ):
                raise _validation_error(
                    "valid_fan_has_no_obstruction",
                    "a VALID fan must not carry an obstruction",
                )
        elif self.obstruction_code is None or self.obstruction_message is None:
            raise _validation_error(
                "invalid_fan_requires_obstruction",
                "an INVALID fan must carry its first obstruction",
            )
        return self

    @classmethod
    def _from_recognition(
        cls, fan: ToricFanPresentation, *, recognized: RecognizedFan
    ) -> Self:
        obstruction = recognized.obstruction
        return cls.model_construct(
            fan=fan,
            status="VALID" if obstruction is None else "INVALID",
            obstruction_code=None if obstruction is None else obstruction.code,
            obstruction_message=None if obstruction is None else obstruction.message,
        )


class OrbitConeRow(StrictModel):
    """One cone's exact orbit-cone profile row."""

    cone_id: ToricRayIndex
    ray_indices: tuple[ToricRayIndex, ...] = Field(max_length=MAX_TORIC_CONE_GENERATORS)
    dimension: ToricRayIndex = Field(description="Rank of the generator span over Q.")
    orbit_dimension: ToricRayIndex = Field(
        description="n - dimension, the dimension of the corresponding torus orbit."
    )
    is_simplicial: StrictBool
    is_smooth: StrictBool = Field(
        description=(
            "Generators are Z-linearly independent and saturated (every Smith "
            "invariant factor is 1)."
        )
    )


class FaceRelation(StrictModel):
    """One orbit-closure incidence ``tau <= sigma``."""

    tau_cone_id: ToricRayIndex
    sigma_cone_id: ToricRayIndex


class RayConeIncidence(StrictModel):
    """All cones containing one fan ray."""

    ray_index: ToricRayIndex
    cone_ids: tuple[ToricRayIndex, ...] = Field(max_length=MAX_TORIC_CONE_COUNT)


class OrbitConeProfileRequest(StrictModel):
    """Compute the orbit-cone profile of one rational fan."""

    fan: ToricFanPresentation = Field(
        description=(
            "Fan presentation re-recognized by the exact kernel before any "
            f"profile row is produced; envelope: {_ENVELOPE_DESCRIPTION}."
        )
    )


class OrbitConeProfileResult(StrictModel):
    """Source-bound orbit-cone correspondence profile of one fan."""

    fan: ToricFanPresentation
    lattice_rank: StrictInt
    cones: tuple[OrbitConeRow, ...] = Field(max_length=MAX_TORIC_CONE_COUNT)
    face_relations: tuple[FaceRelation, ...] = Field(
        description=(
            "Complete pairs tau <= sigma with tau first; the orbit closure of "
            "V(tau) contains V(sigma)."
        )
    )
    ray_incidence: tuple[RayConeIncidence, ...] = Field(max_length=MAX_TORIC_RAY_COUNT)

    @model_validator(mode="after")
    def require_structural_consistency(self) -> Self:
        if tuple(row.cone_id for row in self.cones) != tuple(
            range(len(self.fan.cones))
        ):
            raise _validation_error(
                "orbit_profile_cone_ids",
                "orbit rows must carry consecutive source cone ids",
            )
        if any(
            row.orbit_dimension != self.lattice_rank - row.dimension
            for row in self.cones
        ):
            raise _validation_error(
                "orbit_profile_dimensions",
                "orbit dimension must equal n minus the cone dimension",
            )
        if tuple(row.ray_index for row in self.ray_incidence) != tuple(
            range(len(self.fan.rays))
        ):
            raise _validation_error(
                "orbit_profile_ray_incidence",
                "ray incidence must carry one consecutive row per fan ray",
            )
        if any(
            relation.tau_cone_id > relation.sigma_cone_id
            for relation in self.face_relations
        ):
            raise _validation_error(
                "orbit_profile_face_order",
                "face relations must list the face cone first",
            )
        return self

    @classmethod
    def _from_recognition(
        cls, fan: ToricFanPresentation, *, recognized: RecognizedFan
    ) -> Self:
        incidence = tuple(
            RayConeIncidence.model_construct(
                ray_index=ray_index,
                cone_ids=tuple(
                    cone.cone_id
                    for cone in recognized.cones
                    if ray_index in cone.ray_indices
                ),
            )
            for ray_index in range(len(recognized.rays))
        )
        return cls.model_construct(
            fan=fan,
            lattice_rank=fan.lattice_rank,
            cones=tuple(
                OrbitConeRow.model_construct(
                    cone_id=cone.cone_id,
                    ray_indices=cone.ray_indices,
                    dimension=cone.dimension,
                    orbit_dimension=fan.lattice_rank - cone.dimension,
                    is_simplicial=cone.is_simplicial,
                    is_smooth=cone.is_smooth,
                )
                for cone in recognized.cones
            ),
            face_relations=tuple(
                FaceRelation.model_construct(tau_cone_id=tau, sigma_cone_id=sigma)
                for tau, sigma in recognized.face_relations
            ),
            ray_incidence=incidence,
        )


class CharacterDivisorRequest(StrictModel):
    """Compute div(chi^m) of one character on one fan."""

    fan: ToricFanPresentation = Field(
        description=(
            "Fan presentation re-recognized by the exact kernel before any "
            f"coefficient is produced; envelope: {_ENVELOPE_DESCRIPTION}."
        )
    )
    character: CharacterVector = Field(
        description=(
            "Integer character m in the dual lattice; coefficients are the "
            "exact pairings <m, v_rho>."
        )
    )

    @model_validator(mode="after")
    def require_matching_rank(self) -> Self:
        if len(self.character.entries) != self.fan.lattice_rank:
            raise _validation_error(
                "character_rank_mismatch",
                "the character must have exactly lattice_rank coordinates",
            )
        limit = 10**MAX_TORIC_COORDINATE_DIGITS
        if any(abs(value) >= limit for value in self.character.entries):
            raise _validation_error(
                "coordinates_exceed_envelope",
                "character coordinates exceed the "
                f"{MAX_TORIC_COORDINATE_DIGITS}-digit toric envelope",
            )
        return self


class CharacterDivisorResult(StrictModel):
    """Source-bound principal T-invariant divisor div(chi^m)."""

    fan: ToricFanPresentation
    character: CharacterVector
    coefficients: tuple[ExactInteger, ...] = Field(
        max_length=MAX_TORIC_RAY_COUNT,
        description=(
            "One exact coefficient <m, v_rho> per fan ray, in declared ray "
            "order, zero coefficients retained."
        ),
    )
    is_principal: StrictBool
    support_ray_indices: tuple[ToricRayIndex, ...] = Field(
        max_length=MAX_TORIC_RAY_COUNT,
        description="Rays with a nonzero coefficient, ascending.",
    )

    @model_validator(mode="after")
    def require_structural_consistency(self) -> Self:
        if len(self.coefficients) != len(self.fan.rays):
            raise _validation_error(
                "character_divisor_row_count",
                "the divisor must carry one coefficient per fan ray",
            )
        expected_support = tuple(
            index
            for index, coefficient in enumerate(self.coefficients)
            if coefficient != 0
        )
        if self.support_ray_indices != expected_support:
            raise _validation_error(
                "character_divisor_support",
                "the support must be exactly the nonzero-coefficient rays",
            )
        return self

    @classmethod
    def _from_coefficients(
        cls,
        fan: ToricFanPresentation,
        character: CharacterVector,
        *,
        coefficients: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            fan=fan,
            character=character,
            coefficients=coefficients,
            is_principal=True,
            support_ray_indices=tuple(
                index
                for index, coefficient in enumerate(coefficients)
                if coefficient != 0
            ),
        )


class ToricAffineChartRequest(StrictModel):
    """Compute the affine monomial chart of one full-dimensional fan cone.

    The cone is identified by its strictly increasing ray-index label, which
    must be one of the fan's declared cones. The chart is
    ``Spec k[sigma^vee cap M]``; only full-dimensional cones (and therefore
    pointed dual semigroups with a finite Hilbert basis) are admitted.
    """

    fan: ToricFanPresentation = Field(
        description=(
            "Fan presentation re-recognized by the exact kernel before any "
            f"chart data is produced; envelope: {_ENVELOPE_DESCRIPTION}."
        )
    )
    cone: tuple[ToricRayIndex, ...] = Field(
        max_length=MAX_TORIC_CONE_GENERATORS,
        description=(
            "Strictly increasing ray-index label of a declared full-dimensional "
            "cone of the fan."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_cone_label(self) -> Self:
        if any(first >= second for first, second in itertools.pairwise(self.cone)):
            raise _validation_error(
                "chart_cone_label",
                "the cone label must be strictly increasing ray indices",
            )
        if any(index < 0 or index >= len(self.fan.rays) for index in self.cone):
            raise _validation_error(
                "chart_cone_label", "the cone label must address declared rays"
            )
        return self


class ToricChartLocalization(StrictModel):
    """Restriction of one affine chart to a face cone.

    For a face ``tau <= sigma`` the chart of ``tau`` is the localization
    ``k[S_sigma][1/x^m]`` obtained by adjoining the localizing character ``m``,
    which is zero on the rays of ``tau`` and strictly positive on the remaining
    rays of ``sigma``.
    """

    face_cone_id: ToricRayIndex
    face_ray_indices: tuple[ToricRayIndex, ...] = Field(
        max_length=MAX_TORIC_CONE_GENERATORS
    )
    localizing_character: tuple[ExactInteger, ...] = Field(
        min_length=1, max_length=MAX_TORIC_LATTICE_RANK
    )


class ToricAffineChartResult(StrictModel):
    """Canonical semigroup presentation of one affine monomial chart."""

    fan: ToricFanPresentation
    cone_id: ToricRayIndex
    cone_ray_indices: tuple[ToricRayIndex, ...] = Field(
        max_length=MAX_TORIC_CONE_GENERATORS
    )
    lattice_rank: StrictInt
    dimension: StrictInt
    dual_cone_rays: tuple[tuple[ExactInteger, ...], ...] = Field(
        max_length=MAX_TORIC_CHART_DUAL_RAYS,
        description=(
            "Primitive extreme rays of the dual cone sigma^vee in the dual "
            "lattice M = Z^n, lexicographically sorted."
        ),
    )
    hilbert_basis: tuple[tuple[ExactInteger, ...], ...] = Field(
        max_length=MAX_TORIC_CHART_GENERATORS,
        description=(
            "Complete canonical Hilbert basis of the affine semigroup "
            "sigma^vee cap M, lexicographically sorted."
        ),
    )
    relations: tuple[tuple[ExactInteger, ...], ...] = Field(
        max_length=MAX_TORIC_CHART_RELATIONS,
        description=(
            "Canonical integer relation-lattice basis over the Hilbert basis: "
            "every relation evaluates to zero on the generators."
        ),
    )
    is_smooth: StrictBool
    localizations: tuple[ToricChartLocalization, ...] = Field(
        max_length=MAX_TORIC_CONE_COUNT
    )

    @model_validator(mode="after")
    def require_structural_consistency(self) -> Self:
        if any(len(ray) != self.lattice_rank for ray in self.dual_cone_rays):
            raise _validation_error(
                "chart_dual_cone_rank", "every dual-cone ray must use the lattice"
            )
        if any(len(generator) != self.lattice_rank for generator in self.hilbert_basis):
            raise _validation_error(
                "chart_generator_rank", "every Hilbert generator must use the lattice"
            )
        if any(len(relation) != len(self.hilbert_basis) for relation in self.relations):
            raise _validation_error(
                "chart_relation_width",
                "every relation must carry one coefficient per Hilbert generator",
            )
        if any(
            index < 0 or index >= len(self.fan.rays)
            for index in (*self.cone_ray_indices,)
        ):
            raise _validation_error(
                "chart_cone_label", "the cone label must address declared rays"
            )
        for localization in self.localizations:
            if len(localization.localizing_character) != self.lattice_rank:
                raise _validation_error(
                    "chart_localization_rank",
                    "every localizing character must use the lattice",
                )
        return self

    @classmethod
    def _from_components(
        cls,
        *,
        fan: ToricFanPresentation,
        cone_id: int,
        cone_ray_indices: tuple[int, ...],
        dimension: int,
        dual_cone_rays: tuple[tuple[int, ...], ...],
        hilbert_basis: tuple[tuple[int, ...], ...],
        relations: tuple[tuple[int, ...], ...],
        is_smooth: bool,
        localizations: tuple[ToricChartLocalization, ...],
    ) -> Self:
        return cls.model_construct(
            fan=fan,
            cone_id=cone_id,
            cone_ray_indices=cone_ray_indices,
            lattice_rank=fan.lattice_rank,
            dimension=dimension,
            dual_cone_rays=dual_cone_rays,
            hilbert_basis=hilbert_basis,
            relations=relations,
            is_smooth=is_smooth,
            localizations=localizations,
        )


class ToricMorphismRequest(StrictModel):
    """Decide whether one integer lattice map is a toric morphism."""

    source: ToricFanPresentation = Field(
        description=(
            "Source fan Sigma_1; re-recognized by the exact kernel before any "
            f"membership test. Envelope: {_ENVELOPE_DESCRIPTION}."
        )
    )
    target: ToricFanPresentation = Field(
        description=(
            "Target fan Sigma_2; re-recognized by the exact kernel before any "
            f"membership test. Envelope: {_ENVELOPE_DESCRIPTION}."
        )
    )
    matrix: IntegerMatrix = Field(
        description=(
            "Integer matrix of the lattice map phi: N_1 -> N_2 with "
            "row_count = target lattice rank and column_count = source lattice "
            "rank; phi(v) = matrix . v."
        )
    )

    @model_validator(mode="after")
    def require_matching_lattices(self) -> Self:
        if self.matrix.row_count != self.target.lattice_rank:
            raise _validation_error(
                "morphism_matrix_rows",
                "the matrix must have exactly target lattice rank rows",
            )
        if self.matrix.column_count != self.source.lattice_rank:
            raise _validation_error(
                "morphism_matrix_columns",
                "the matrix must have exactly source lattice rank columns",
            )
        limit = 10**MAX_TORIC_COORDINATE_DIGITS
        if any(
            abs(int(value)) >= limit for row in self.matrix.entries for value in row
        ):
            raise _validation_error(
                "morphism_matrix_digits",
                "matrix entries exceed the "
                f"{MAX_TORIC_COORDINATE_DIGITS}-digit toric envelope",
            )
        return self


class ToricConeAssignment(StrictModel):
    """One induced fan-compatible source-cone to target-cone assignment."""

    source_cone_id: ToricRayIndex
    target_cone_id: ToricRayIndex


class ToricMorphismObstruction(StrictModel):
    """The first source cone whose image leaves every target cone."""

    source_cone_id: ToricRayIndex
    source_ray_indices: tuple[ToricRayIndex, ...] = Field(
        max_length=MAX_TORIC_CONE_GENERATORS
    )
    image_vectors: tuple[tuple[ExactInteger, ...], ...] = Field(
        max_length=MAX_TORIC_CONE_GENERATORS
    )
    candidate_target_cone_ids: tuple[tuple[ToricRayIndex, ...], ...] = Field(
        max_length=MAX_TORIC_CONE_GENERATORS,
        description=(
            "Target cones containing each image ray; a ray image outside every "
            "target cone has an empty entry."
        ),
    )
    failing_ray_index: ToricRayIndex | None = None
    failing_image: tuple[ExactInteger, ...] | None = None
    reason: Literal[
        "RAY_IMAGE_OUTSIDE_ALL_TARGET_CONES",
        "NO_COMMON_TARGET_CONE",
    ]

    @model_validator(mode="after")
    def require_bound_witness(self) -> Self:
        if self.reason == "RAY_IMAGE_OUTSIDE_ALL_TARGET_CONES":
            if self.failing_ray_index is None or self.failing_image is None:
                raise _validation_error(
                    "morphism_obstruction_witness",
                    "a ray-outside witness requires the failing ray and image",
                )
        elif self.failing_ray_index is not None or self.failing_image is not None:
            raise _validation_error(
                "morphism_obstruction_witness",
                "a no-common-cone obstruction carries no single failing ray",
            )
        return self


class ToricMorphismResult(StrictModel):
    """Toric-morphism verdict with the induced fan-compatible cone map."""

    source: ToricFanPresentation
    target: ToricFanPresentation
    matrix: IntegerMatrix
    is_toric_morphism: StrictBool
    ray_images: tuple[tuple[ExactInteger, ...], ...] = Field(
        max_length=MAX_TORIC_RAY_COUNT,
        description="Image phi(v_rho) of every source ray, in declared order.",
    )
    assignments: tuple[ToricConeAssignment, ...] = Field(
        max_length=MAX_TORIC_CONE_COUNT
    )
    obstruction: ToricMorphismObstruction | None = None

    @model_validator(mode="after")
    def require_verdict_shape(self) -> Self:
        if self.is_toric_morphism:
            if self.obstruction is not None:
                raise _validation_error(
                    "morphism_verdict",
                    "a toric morphism must not carry an obstruction",
                )
            if len(self.assignments) != len(self.source.cones):
                raise _validation_error(
                    "morphism_verdict",
                    "a toric morphism must assign every source cone",
                )
        elif self.obstruction is None:
            raise _validation_error(
                "morphism_verdict",
                "a non-morphism must carry its first obstruction",
            )
        return self

    @classmethod
    def _from_components(
        cls,
        *,
        source: ToricFanPresentation,
        target: ToricFanPresentation,
        matrix: IntegerMatrix,
        is_toric_morphism: bool,
        ray_images: tuple[tuple[int, ...], ...],
        assignments: tuple[ToricConeAssignment, ...],
        obstruction: ToricMorphismObstruction | None,
    ) -> Self:
        return cls.model_construct(
            source=source,
            target=target,
            matrix=matrix,
            is_toric_morphism=is_toric_morphism,
            ray_images=ray_images,
            assignments=assignments,
            obstruction=obstruction,
        )


__all__ = [
    "MAX_TORIC_CHART_BOX",
    "MAX_TORIC_CHART_CANDIDATES",
    "MAX_TORIC_CHART_COMPONENT_DIGITS",
    "MAX_TORIC_CHART_DUAL_RAYS",
    "MAX_TORIC_CHART_GENERATORS",
    "MAX_TORIC_CHART_RELATIONS",
    "MAX_TORIC_CONE_COUNT",
    "MAX_TORIC_CONE_GENERATORS",
    "MAX_TORIC_COORDINATE_DIGITS",
    "MAX_TORIC_LATTICE_RANK",
    "MAX_TORIC_MORPHISM_WORK",
    "MAX_TORIC_RAY_COUNT",
    "CharacterDivisorRequest",
    "CharacterDivisorResult",
    "CharacterVector",
    "FaceRelation",
    "FanValidationRequest",
    "FanValidationResult",
    "OrbitConeProfileRequest",
    "OrbitConeProfileResult",
    "OrbitConeRow",
    "RayConeIncidence",
    "ToricAffineChartRequest",
    "ToricAffineChartResult",
    "ToricChartLocalization",
    "ToricConeAssignment",
    "ToricFanPresentation",
    "ToricMorphismObstruction",
    "ToricMorphismRequest",
    "ToricMorphismResult",
]
