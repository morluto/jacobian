"""Envelope accept/reject boundary tests for the toric operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from tests.math.geometry.toric._fixtures import fan, p4_fan

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.geometry.toric._models import (
    MAX_TORIC_CONE_COUNT,
    MAX_TORIC_CONE_GENERATORS,
    MAX_TORIC_COORDINATE_DIGITS,
    MAX_TORIC_LATTICE_RANK,
    MAX_TORIC_RAY_COUNT,
    ToricFanPresentation,
)
from jacobian.math.geometry.toric._tools import TOOLS
from jacobian.math.geometry.toric.operations import (
    compute_character_divisor,
    compute_orbit_cone_profile,
    validate_fan,
)

TWELVE_RAY_CIRCLE = (
    (1, 0),
    (2, 1),
    (1, 1),
    (1, 2),
    (0, 1),
    (-1, 2),
    (-1, 1),
    (-2, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
)

OCTAGON_RAYS = (
    (1, 2, 1),
    (1, 1, 2),
    (1, -1, 2),
    (1, -2, 1),
    (1, -2, -1),
    (1, -1, -2),
    (1, 1, -2),
    (1, 2, -1),
)


def _twelve_ray_fan() -> ToricFanPresentation:
    cones: list[tuple[int, ...]] = [()]
    cones.extend((index,) for index in range(12))
    cones.extend((index, index + 1) for index in range(11))
    cones.append((0, 11))
    return fan(TWELVE_RAY_CIRCLE, tuple(cones))


def _octagon_fan() -> ToricFanPresentation:
    cones: list[tuple[int, ...]] = [()]
    cones.extend((index,) for index in range(8))
    cones.extend((index, index + 1) for index in range(7))
    cones.append((0, 7))
    cones.append(tuple(range(8)))
    return fan(OCTAGON_RAYS, tuple(cones), rank=3)


def test_rank_four_is_accepted() -> None:
    assert validate_fan(p4_fan()).status == "VALID"
    assert MAX_TORIC_LATTICE_RANK == 4


def test_twelve_rays_are_accepted() -> None:
    presentation = _twelve_ray_fan()
    assert len(presentation.rays) == MAX_TORIC_RAY_COUNT
    assert len(presentation.cones) == 25
    result = validate_fan(presentation)
    assert result.status == "VALID"
    profile = compute_orbit_cone_profile(presentation)
    assert sum(1 for row in profile.cones if row.dimension == 2) == 12
    assert all(row.is_smooth for row in profile.cones)


def test_thirteen_rays_are_rejected_on_the_wire() -> None:
    with pytest.raises(ValidationError):
        fan((*TWELVE_RAY_CIRCLE, (3, 1)), ((),))


def test_octagon_cone_with_eight_generators_is_accepted() -> None:
    presentation = _octagon_fan()
    assert any(len(cone) == MAX_TORIC_CONE_GENERATORS for cone in presentation.cones)
    assert validate_fan(presentation).status == "VALID"
    profile = compute_orbit_cone_profile(presentation)
    top = max(profile.cones, key=lambda row: row.dimension)
    assert top.dimension == 3
    assert top.is_simplicial is False
    assert top.is_smooth is False


def test_nine_generators_are_rejected_on_the_wire() -> None:
    with pytest.raises(ValidationError):
        fan((*OCTAGON_RAYS, (5, 1, 1)), ((), tuple(range(9))), rank=3)


def test_forty_one_cones_are_rejected_on_the_wire() -> None:
    cones = [()] + [
        (first, second) for first in range(12) for second in range(first + 1, 12)
    ]
    assert len(cones) > MAX_TORIC_CONE_COUNT
    with pytest.raises(ValidationError):
        fan(TWELVE_RAY_CIRCLE, tuple(cones))


def test_coordinate_digit_boundary_is_accepted() -> None:
    largest = 10**MAX_TORIC_COORDINATE_DIGITS - 1
    presentation = fan(((largest, 1), (1, 0)), ((), (0,), (1,), (0, 1)))
    assert validate_fan(presentation).status == "VALID"


def test_coordinate_digit_boundary_is_rejected_on_the_wire() -> None:
    oversized = 10**MAX_TORIC_COORDINATE_DIGITS
    with pytest.raises(ValidationError):
        fan(((oversized, 1), (1, 0)), ((), (0,), (1,), (0, 1)))


def test_native_admission_rejects_constructed_over_envelope_values() -> None:
    from jacobian.math.geometry.toric._models import CharacterVector

    over_rank = ToricFanPresentation.model_construct(
        lattice_rank=MAX_TORIC_LATTICE_RANK + 1, rays=(), cones=()
    )
    over_character = CharacterVector.model_construct(entries=(0,) * 5)
    with pytest.raises(OperationResourceAdmissionError):
        compute_character_divisor(over_rank, over_character)
    over_rays = ToricFanPresentation.model_construct(
        lattice_rank=2,
        rays=tuple((index, 1) for index in range(MAX_TORIC_RAY_COUNT + 1)),
        cones=(),
    )
    with pytest.raises(OperationResourceAdmissionError):
        compute_orbit_cone_profile(over_rays)
    over_generators = ToricFanPresentation.model_construct(
        lattice_rank=3,
        rays=(*OCTAGON_RAYS, (5, 1, 1)),
        cones=(tuple(range(MAX_TORIC_CONE_GENERATORS + 1)),),
    )
    with pytest.raises(OperationResourceAdmissionError):
        validate_fan(over_generators)
    over_cones = ToricFanPresentation.model_construct(
        lattice_rank=2,
        rays=TWELVE_RAY_CIRCLE,
        cones=tuple(
            (first, second) for first in range(12) for second in range(first + 1, 12)
        ),
    )
    with pytest.raises(OperationResourceAdmissionError):
        validate_fan(over_cones)


def test_envelope_constants_are_published_in_every_description() -> None:
    for tool in TOOLS:
        assert str(MAX_TORIC_LATTICE_RANK) in tool.description
        assert str(MAX_TORIC_RAY_COUNT) in tool.description
        assert str(MAX_TORIC_CONE_COUNT) in tool.description
        assert str(MAX_TORIC_CONE_GENERATORS) in tool.description
        assert str(MAX_TORIC_COORDINATE_DIGITS) in tool.description
