"""Native exact toric-geometry operations."""

from __future__ import annotations

from math import gcd
from typing import NoReturn

from jacobian._execution import execution_deadline
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.toric._kernel import (
    RecognizedCone,
    RecognizedFan,
    _integer_kernel_basis,
    _lift_quotient_character,
    _quotient_projection,
    _ray_in_cone,
    _unimodular_complement,
    compute_affine_chart_data,
    compute_toric_morphism_data,
    facet_localizing_character,
    recognize_fan,
)
from jacobian.math.geometry.toric._models import (
    MAX_TORIC_CONE_COUNT,
    MAX_TORIC_CONE_GENERATORS,
    MAX_TORIC_COORDINATE_DIGITS,
    MAX_TORIC_LATTICE_RANK,
    MAX_TORIC_MORPHISM_WORK,
    MAX_TORIC_RAY_COUNT,
    CharacterDivisorResult,
    CharacterVector,
    FanValidationResult,
    OrbitConeProfileResult,
    ToricAffineChartResult,
    ToricChartLocalization,
    ToricConeAssignment,
    ToricFanPresentation,
    ToricMorphismObstruction,
    ToricMorphismResult,
)
from jacobian.math.matrices.values import IntegerMatrix

MAX_TORIC_RECOGNITION_WORK = 5_000_000

# Operation-owned wall allowance for the affine-chart and morphism kernels. The
# admitted envelopes keep every mandatory phase well inside this margin; the
# deadline is a safety net for the untrusted Z3 reduction, never mathematical
# evidence.
MAX_TORIC_CHART_SECONDS = 30.0
MAX_TORIC_MORPHISM_SECONDS = 30.0


def _reject_envelope(message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=("fan",),
        code="toric.resource_budget_exceeded",
        message=message,
    )


def _reject_domain(
    location: tuple[str | int, ...], code: str, message: str
) -> NoReturn:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def _admit_fan(fan: ToricFanPresentation) -> None:
    """Enforce the published envelope and preflight the recognition work.

    Catalog requests are bounded by the presentation model; native callers can
    bypass wire validation, so the same envelope is re-checked here, together
    with exact preflight counts for subset enumeration, membership problems,
    and result rows, before any recognition arithmetic starts.
    """

    if not 1 <= fan.lattice_rank <= MAX_TORIC_LATTICE_RANK:
        _reject_envelope(f"lattice rank is limited to {MAX_TORIC_LATTICE_RANK}")
    if len(fan.rays) > MAX_TORIC_RAY_COUNT:
        _reject_envelope(f"fans are limited to {MAX_TORIC_RAY_COUNT} rays")
    if len(fan.cones) > MAX_TORIC_CONE_COUNT:
        _reject_envelope(f"fans are limited to {MAX_TORIC_CONE_COUNT} cones")
    limit = 10**MAX_TORIC_COORDINATE_DIGITS
    for ray in fan.rays:
        if len(ray) != fan.lattice_rank:
            _reject_domain(
                ("fan", "rays"),
                "toric.fan_presentation_malformed",
                "every ray must have exactly lattice_rank coordinates",
            )
        if any(abs(value) >= limit for value in ray):
            _reject_envelope(
                "ray coordinates are limited to "
                f"{MAX_TORIC_COORDINATE_DIGITS} decimal digits"
            )
    face_subsets = 0
    membership_problems = 0
    for cone in fan.cones:
        if len(cone) > MAX_TORIC_CONE_GENERATORS:
            _reject_envelope(
                f"cones are limited to {MAX_TORIC_CONE_GENERATORS} generators"
            )
        if any(index < 0 or index >= len(fan.rays) for index in cone):
            _reject_domain(
                ("fan", "cones"),
                "toric.fan_presentation_malformed",
                "cone ray indices must address declared rays",
            )
        face_subsets += 2 ** len(cone)
        membership_problems += len(cone)
    cone_count = len(fan.cones)
    pair_problems = cone_count * (cone_count - 1) // 2
    membership_problems += pair_problems * 2 * MAX_TORIC_CONE_GENERATORS
    recognition_work = (
        face_subsets + membership_problems + cone_count * cone_count
    ) * (fan.lattice_rank + MAX_TORIC_CONE_GENERATORS)
    if recognition_work > MAX_TORIC_RECOGNITION_WORK:
        _reject_envelope(f"fan recognition work exceeds {MAX_TORIC_RECOGNITION_WORK}")


def _admit_character(fan: ToricFanPresentation, character: CharacterVector) -> None:
    if len(character.entries) != fan.lattice_rank:
        _reject_domain(
            ("character",),
            "toric.character_rank_mismatch",
            "the character must have exactly lattice_rank coordinates",
        )
    limit = 10**MAX_TORIC_COORDINATE_DIGITS
    if any(abs(value) >= limit for value in character.entries):
        _reject_envelope(
            "character coordinates are limited to "
            f"{MAX_TORIC_COORDINATE_DIGITS} decimal digits"
        )


def _recognize_or_reject(fan: ToricFanPresentation) -> RecognizedFan:
    """Re-recognize a caller-authored fan claim once and reuse its facts."""

    _admit_fan(fan)
    recognized = recognize_fan(fan.rays, fan.cones, fan.lattice_rank)
    if recognized.obstruction is not None:
        _reject_domain(
            ("fan",),
            "toric.fan_not_valid",
            f"{recognized.obstruction.code}: {recognized.obstruction.message}",
        )
    return recognized


def validate_fan(fan: ToricFanPresentation) -> FanValidationResult:
    """Recognize a proposed rational fan and retain the first obstruction."""

    _admit_fan(fan)
    recognized = recognize_fan(fan.rays, fan.cones, fan.lattice_rank)
    return FanValidationResult._from_recognition(fan, recognized=recognized)


def compute_orbit_cone_profile(fan: ToricFanPresentation) -> OrbitConeProfileResult:
    """Return the source-bound orbit-cone profile of one validated fan."""

    recognized = _recognize_or_reject(fan)
    return OrbitConeProfileResult._from_recognition(fan, recognized=recognized)


def compute_character_divisor(
    fan: ToricFanPresentation, character: CharacterVector
) -> CharacterDivisorResult:
    """Return div(chi^m) = sum_rho <m, v_rho> D_rho exactly."""

    recognized = _recognize_or_reject(fan)
    _admit_character(fan, character)
    coefficients = tuple(
        sum(
            m_value * ray[coordinate]
            for coordinate, m_value in enumerate(character.entries)
        )
        for ray in recognized.rays
    )
    return CharacterDivisorResult._from_coefficients(
        fan, character, coefficients=coefficients
    )


def _primitive_character(values: tuple[int, ...]) -> tuple[int, ...]:
    divisor = 0
    for value in values:
        divisor = gcd(divisor, abs(value))
    if divisor <= 1:
        return values
    return tuple(value // divisor for value in values)


def _locate_cone(recognized: RecognizedFan, cone: tuple[int, ...]) -> RecognizedCone:
    label = tuple(sorted(set(cone)))
    for recognized_cone in recognized.cones:
        if recognized_cone.ray_indices == label:
            return recognized_cone
    _reject_domain(
        ("cone",),
        "toric.chart_cone_not_declared",
        "the cone label must be one of the fan's declared cones",
    )


def _chart_localizations(
    recognized: RecognizedFan, cone_id: int, lattice_rank: int
) -> tuple[ToricChartLocalization, ...]:
    """Localizing characters for every proper face of one cone.

    For a full-dimensional cone the characters are the facet characters in
    the ambient lattice.  For a lower-dimensional cone they are computed on
    the pointed quotient ``sigma^vee / sigma^perp`` and lifted along the
    quotient projection, so ``<m, r> = <m', pi(r)>`` on every ray.
    """

    sigma = recognized.cones[cone_id]
    sigma_rays = tuple(recognized.rays[index] for index in sigma.ray_indices)
    quotient = sigma.dimension < lattice_rank
    projection: tuple[tuple[int, ...], ...] = ()
    quotient_dimension = sigma.dimension
    if quotient:
        lineality = _integer_kernel_basis(sigma_rays, lattice_rank)
        complement = _unimodular_complement(lineality, lattice_rank)
        projection = _quotient_projection(
            lineality, complement, lattice_rank, quotient_dimension
        )

    def quotient_ray(ray: tuple[int, ...]) -> tuple[int, ...]:
        if not quotient:
            return ray
        return tuple(
            sum(row[index] * ray[index] for index in range(lattice_rank))
            for row in projection
        )

    sigma_quotient_rays = tuple(quotient_ray(ray) for ray in sigma_rays)
    facet_characters: dict[int, tuple[int, ...]] = {}
    for facet in recognized.cones:
        if (
            facet.cone_id == cone_id
            or facet.dimension != sigma.dimension - 1
            or not set(facet.ray_indices).issubset(sigma.ray_indices)
        ):
            continue
        facet_rays = tuple(
            quotient_ray(recognized.rays[index]) for index in facet.ray_indices
        )
        character = facet_localizing_character(
            sigma_quotient_rays, facet_rays, quotient_dimension
        )
        if character is None:
            raise ArithmeticError(
                "a declared facet of a cone has no primitive supporting character"
            )
        facet_characters[facet.cone_id] = (
            _lift_quotient_character(character, projection) if quotient else character
        )
    localizations: list[ToricChartLocalization] = []
    for tau_id, sigma_id in recognized.face_relations:
        if sigma_id != cone_id or tau_id == cone_id:
            continue
        tau = recognized.cones[tau_id]
        tau_rays = set(tau.ray_indices)
        total = [0] * lattice_rank
        for facet in recognized.cones:
            if facet.cone_id not in facet_characters:
                continue
            if not tau_rays.issubset(facet.ray_indices):
                continue
            character = facet_characters[facet.cone_id]
            for coordinate in range(lattice_rank):
                total[coordinate] += character[coordinate]
        localizations.append(
            ToricChartLocalization.model_construct(
                face_cone_id=tau_id,
                face_ray_indices=tau.ray_indices,
                localizing_character=_primitive_character(tuple(total)),
            )
        )
    localizations.sort(key=lambda localization: localization.face_cone_id)
    return tuple(localizations)


def compute_affine_chart(
    fan: ToricFanPresentation, cone: tuple[int, ...]
) -> ToricAffineChartResult:
    """Return the affine chart ``Spec k[sigma^vee cap M]`` of one fan cone.

    The cone is resolved exactly.  A full-dimensional cone uses the shipped
    double description; a lower-dimensional cone is decomposed into its free
    torus lineality factor and the pointed quotient semigroup, and its facet
    localizations are computed on that quotient.  The complete Hilbert basis
    is reduced from the fundamental-parallelepiped candidates and the relation
    lattice is replayed against the generators.
    """

    recognized = _recognize_or_reject(fan)
    recognized_cone = _locate_cone(recognized, cone)
    deadline = execution_deadline(MAX_TORIC_CHART_SECONDS)
    data = compute_affine_chart_data(
        tuple(recognized.rays[index] for index in recognized_cone.ray_indices),
        fan.lattice_rank,
        deadline,
    )
    localizations = _chart_localizations(
        recognized, recognized_cone.cone_id, fan.lattice_rank
    )
    return ToricAffineChartResult._from_components(
        fan=fan,
        cone_id=recognized_cone.cone_id,
        cone_ray_indices=recognized_cone.ray_indices,
        dimension=recognized_cone.dimension,
        dual_cone_rays=data.dual_cone_rays,
        hilbert_basis=data.hilbert_basis,
        relations=data.relations,
        torus_basis=data.torus_basis,
        is_smooth=recognized_cone.is_smooth,
        localizations=localizations,
    )


def _admit_morphism_matrix(
    source: ToricFanPresentation,
    target: ToricFanPresentation,
    matrix: IntegerMatrix,
) -> None:
    entries = matrix.entries
    if len(entries) != target.lattice_rank or any(
        len(row) != source.lattice_rank for row in entries
    ):
        _reject_domain(
            ("matrix",),
            "toric.morphism_matrix_shape",
            "the matrix must have target lattice rank rows and source lattice "
            "rank columns",
        )
    limit = 10**MAX_TORIC_COORDINATE_DIGITS
    if any(abs(int(value)) >= limit for row in entries for value in row):
        _reject_envelope(
            "morphism matrix entries are limited to "
            f"{MAX_TORIC_COORDINATE_DIGITS} decimal digits"
        )


def check_toric_morphism(
    source: ToricFanPresentation,
    target: ToricFanPresentation,
    matrix: IntegerMatrix,
) -> ToricMorphismResult:
    """Decide whether one integer lattice map is a toric morphism.

    Every source cone must map into a single target cone. The verdict returns
    the induced fan-compatible cone assignment, or the first source cone whose
    image leaves every target cone together with the witnessing ray images. Each
    assignment is replayed with exact cone-membership checks.
    """

    recognized_source = _recognize_or_reject(source)
    recognized_target = _recognize_or_reject(target)
    _admit_morphism_matrix(source, target, matrix)
    source_cones = tuple(cone.ray_indices for cone in recognized_source.cones)
    target_cones = tuple(cone.ray_indices for cone in recognized_target.cones)
    work = sum(len(cone) + 1 for cone in source_cones) * len(target_cones)
    if work > MAX_TORIC_MORPHISM_WORK:
        _reject_envelope(
            f"morphism cone-membership work exceeds {MAX_TORIC_MORPHISM_WORK}"
        )
    execution_deadline(MAX_TORIC_MORPHISM_SECONDS)
    data = compute_toric_morphism_data(
        recognized_source.rays,
        source_cones,
        recognized_target.rays,
        target_cones,
        matrix.entries,
    )
    if data.is_toric_morphism:
        for source_cone_id, target_cone_id in data.assignments:
            source_cone = source_cones[source_cone_id]
            target_generators = tuple(
                recognized_target.rays[index] for index in target_cones[target_cone_id]
            )
            for ray_index in source_cone:
                if not _ray_in_cone(data.ray_images[ray_index], target_generators):
                    raise ArithmeticError(
                        "assigned target cone does not contain every ray image"
                    )
        return ToricMorphismResult._from_components(
            source=source,
            target=target,
            matrix=matrix,
            is_toric_morphism=True,
            ray_images=data.ray_images,
            assignments=tuple(
                ToricConeAssignment.model_construct(
                    source_cone_id=source_cone_id,
                    target_cone_id=target_cone_id,
                )
                for source_cone_id, target_cone_id in data.assignments
            ),
            obstruction=None,
        )
    obstruction = data.obstruction
    if obstruction is None:
        raise ArithmeticError("a non-morphism must carry its first obstruction")
    for position, candidates in enumerate(obstruction.candidate_target_cone_ids):
        image = obstruction.image_vectors[position]
        for target_cone_id in candidates:
            target_generators = tuple(
                recognized_target.rays[index] for index in target_cones[target_cone_id]
            )
            if not _ray_in_cone(image, target_generators):
                raise ArithmeticError(
                    "declared candidate target cone does not contain the ray image"
                )
    return ToricMorphismResult._from_components(
        source=source,
        target=target,
        matrix=matrix,
        is_toric_morphism=False,
        ray_images=data.ray_images,
        assignments=(),
        obstruction=ToricMorphismObstruction.model_construct(
            source_cone_id=obstruction.source_cone_id,
            source_ray_indices=obstruction.source_ray_indices,
            image_vectors=obstruction.image_vectors,
            candidate_target_cone_ids=obstruction.candidate_target_cone_ids,
            failing_ray_index=obstruction.failing_ray_index,
            failing_image=obstruction.failing_image,
            reason=obstruction.reason,
        ),
    )


__all__ = [
    "MAX_TORIC_CHART_SECONDS",
    "MAX_TORIC_MORPHISM_SECONDS",
    "MAX_TORIC_RECOGNITION_WORK",
    "check_toric_morphism",
    "compute_affine_chart",
    "compute_character_divisor",
    "compute_orbit_cone_profile",
    "validate_fan",
]
