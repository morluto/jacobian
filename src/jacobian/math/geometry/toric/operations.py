"""Native exact toric-geometry operations."""

from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.toric._kernel import RecognizedFan, recognize_fan
from jacobian.math.geometry.toric._models import (
    MAX_TORIC_CONE_COUNT,
    MAX_TORIC_CONE_GENERATORS,
    MAX_TORIC_COORDINATE_DIGITS,
    MAX_TORIC_LATTICE_RANK,
    MAX_TORIC_RAY_COUNT,
    CharacterDivisorResult,
    CharacterVector,
    FanValidationResult,
    OrbitConeProfileResult,
    ToricFanPresentation,
)

MAX_TORIC_RECOGNITION_WORK = 5_000_000


def _reject_envelope(message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("fan",),
        code="toric.resource_budget_exceeded",
        message=message,
    )


def _reject_domain(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=location, code=code, message=message
    )


def _admit_fan(fan: ToricFanPresentation) -> None:
    """Enforce the published envelope and preflight the recognition work.

    Catalog requests are bounded by the presentation model; native callers can
    bypass wire validation, so the same envelope is re-checked here, together
    with exact preflight counts for subset enumeration, membership problems,
    and result rows, before any recognition arithmetic starts.
    """

    if not 1 <= fan.lattice_rank <= MAX_TORIC_LATTICE_RANK:
        _reject_envelope(
            f"lattice rank is limited to {MAX_TORIC_LATTICE_RANK}"
        )
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
                "cones are limited to "
                f"{MAX_TORIC_CONE_GENERATORS} generators"
            )
        if any(index < 0 or index >= len(fan.rays) for index in cone):
            _reject_domain(
                ("fan", "cones"),
                "toric.fan_presentation_malformed",
                "cone ray indices must address declared rays",
            )
        face_subsets += 2**len(cone)
        membership_problems += len(cone)
    cone_count = len(fan.cones)
    pair_problems = cone_count * (cone_count - 1) // 2
    membership_problems += pair_problems * 2 * MAX_TORIC_CONE_GENERATORS
    recognition_work = (
        face_subsets + membership_problems + cone_count * cone_count
    ) * (fan.lattice_rank + MAX_TORIC_CONE_GENERATORS)
    if recognition_work > MAX_TORIC_RECOGNITION_WORK:
        _reject_envelope(
            f"fan recognition work exceeds {MAX_TORIC_RECOGNITION_WORK}"
        )


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
        sum(m_value * ray[coordinate] for coordinate, m_value in enumerate(character.entries))
        for ray in recognized.rays
    )
    return CharacterDivisorResult._from_coefficients(
        fan, character, coefficients=coefficients
    )


__all__ = [
    "MAX_TORIC_RECOGNITION_WORK",
    "compute_character_divisor",
    "compute_orbit_cone_profile",
    "validate_fan",
]
