"""Exact recognition tests for proposed rational fan presentations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from tests.math.geometry.toric._fixtures import (
    A2_CONES,
    A2_RAYS,
    DP6_RAYS,
    P2_RAYS,
    P4_RAYS,
    SINGULAR_CONES,
    SQUARE_CONES,
    SQUARE_RAYS,
    a2_fan,
    dp6_fan,
    fan,
    p2_fan,
    p4_fan,
    singular_fan,
    square_fan,
)

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import parse_operation_input
from jacobian.math.geometry.toric._models import (
    FanValidationRequest,
    FanValidationResult,
    ToricFanPresentation,
)
from jacobian.math.geometry.toric._tools import FAN_VALIDATE_OPERATION
from jacobian.math.geometry.toric.operations import validate_fan


@pytest.mark.parametrize(
    "presentation",
    [a2_fan(), p2_fan(), dp6_fan(), singular_fan(), square_fan(), p4_fan()],
    ids=["affine_plane", "projective_plane", "del_pezzo_six", "singular_chart",
         "square_pyramid", "projective_four_space"],
)
def test_known_fans_are_recognized_valid(presentation: ToricFanPresentation) -> None:
    result = validate_fan(presentation)
    assert result.status == "VALID"
    assert result.obstruction_code is None
    assert result.fan == presentation


def test_zero_cone_only_fan_is_valid() -> None:
    result = validate_fan(fan((), ((),), rank=3))
    assert result.status == "VALID"


def test_single_ray_fan_is_valid() -> None:
    result = validate_fan(fan(((1, 0),), ((), (0,)), rank=2))
    assert result.status == "VALID"


def test_missing_face_is_an_exact_obstruction() -> None:
    cones = tuple(cone for cone in SQUARE_CONES if cone != (0, 1))
    result = validate_fan(fan(SQUARE_RAYS, cones, rank=3))
    assert result.status == "INVALID"
    assert result.obstruction_code == "toric.missing_face_cone"


def test_non_primitive_ray_is_rejected() -> None:
    result = validate_fan(
        fan(((2, 0), (0, 1)), ((), (0,), (1,), (0, 1)))
    )
    assert result.status == "INVALID"
    assert result.obstruction_code == "toric.ray_not_primitive"


def test_zero_ray_is_rejected() -> None:
    result = validate_fan(fan(((0, 0), (1, 0)), ((), (0,), (1,),)))
    assert result.status == "INVALID"
    assert result.obstruction_code == "toric.ray_zero"


def test_duplicate_rays_are_rejected() -> None:
    result = validate_fan(
        fan(((1, 0), (1, 0)), ((), (0,), (1,), (0, 1)))
    )
    assert result.status == "INVALID"
    assert result.obstruction_code == "toric.duplicate_rays"


def test_cone_containing_a_line_is_rejected() -> None:
    result = validate_fan(
        fan(((1, 0), (-1, 0)), ((), (0,), (1,), (0, 1)))
    )
    assert result.status == "INVALID"
    assert result.obstruction_code == "toric.cone_not_strongly_convex"


def test_non_extreme_declared_generator_is_rejected() -> None:
    rays = ((1, 0), (0, 1), (1, 1))
    cones = ((), (0,), (1,), (2,), (0, 1, 2))
    result = validate_fan(fan(rays, cones))
    assert result.status == "INVALID"
    assert result.obstruction_code == "toric.generator_not_extreme"


def test_ray_without_a_one_dimensional_cone_is_rejected() -> None:
    result = validate_fan(fan(A2_RAYS, ((), (0,), (0, 1))))
    assert result.status == "INVALID"
    assert result.obstruction_code == "toric.ray_not_a_cone"


def test_intersection_that_is_not_a_common_face_is_rejected() -> None:
    # The ray (1,1) lies in the interior of cone(e1,e2), so the two cones
    # intersect in a ray that is not a face of the quadrant cone.
    rays = ((1, 0), (0, 1), (1, 1))
    cones = ((), (0,), (1,), (2,), (0, 1))
    result = validate_fan(fan(rays, cones))
    assert result.status == "INVALID"
    assert result.obstruction_code == "toric.intersection_not_common_face"


def test_duplicate_cones_are_structurally_rejected() -> None:
    with pytest.raises(ValidationError):
        fan(A2_RAYS, ((), (0,), (0,), (1,), (0, 1)))


def test_unsorted_cone_indices_are_structurally_rejected() -> None:
    with pytest.raises(ValidationError):
        fan(A2_RAYS, ((), (0,), (1,), (1, 0)))


def test_out_of_range_cone_indices_are_structurally_rejected() -> None:
    with pytest.raises(ValidationError):
        fan(A2_RAYS, ((), (0,), (1,), (0, 2)))


def test_ray_length_must_match_lattice_rank() -> None:
    with pytest.raises(ValidationError):
        fan(((1, 0, 0),), ((), (0,)), rank=2)


def test_validation_result_round_trips_through_strict_json() -> None:
    result = validate_fan(p2_fan())
    replayed = FanValidationResult.model_validate_json(result.model_dump_json())
    assert replayed == result
    invalid = validate_fan(fan(A2_RAYS, ((), (0,), (0, 1))))
    replayed_invalid = FanValidationResult.model_validate_json(
        invalid.model_dump_json()
    )
    assert replayed_invalid == invalid
    assert replayed_invalid.fan == invalid.fan


def test_catalog_and_native_paths_share_one_recognition() -> None:
    payload = {
        "fan": {
            "lattice_rank": 2,
            "rays": [["1", "0"], ["0", "1"], ["-1", "-1"]],
            "cones": [[], [0], [1], [2], [0, 1], [0, 2], [1, 2]],
        }
    }
    request = parse_operation_input(FanValidationRequest, payload)
    catalog_result = FAN_VALIDATE_OPERATION.run(request)
    native_result = validate_fan(p2_fan())
    assert catalog_result == native_result


def test_native_envelope_rejects_an_oversized_constructed_presentation() -> None:
    oversized = ToricFanPresentation.model_construct(
        lattice_rank=2,
        rays=tuple((index, 1) for index in range(13)),
        cones=(),
    )
    with pytest.raises(OperationResourceAdmissionError) as rejection:
        validate_fan(oversized)
    assert rejection.value.errors()[0]["type"] == "toric.resource_budget_exceeded"


def test_native_envelope_rejects_oversized_coordinates() -> None:
    oversized = ToricFanPresentation.model_construct(
        lattice_rank=2,
        rays=((10**9, 1), (1, 0)),
        cones=(),
    )
    with pytest.raises(OperationResourceAdmissionError):
        validate_fan(oversized)


def test_native_envelope_rejects_high_rank() -> None:
    oversized = ToricFanPresentation.model_construct(
        lattice_rank=5,
        rays=(),
        cones=(),
    )
    with pytest.raises(OperationResourceAdmissionError):
        validate_fan(oversized)


def test_wire_envelope_rejects_nine_digit_coordinates() -> None:
    with pytest.raises(ValidationError):
        fan(((123456789, 1), (1, 0)), A2_CONES)


def test_known_ray_and_cone_counts_are_retained() -> None:
    assert len(p2_fan().rays) == 3 and len(p2_fan().cones) == 7
    assert len(dp6_fan().rays) == 6 and len(dp6_fan().cones) == 13
    assert len(a2_fan().rays) == 2 and len(a2_fan().cones) == 4
    assert len(singular_fan().rays) == 2
    assert len(p4_fan().cones) == 31
    assert P2_RAYS == ((1, 0), (0, 1), (-1, -1))
    assert DP6_RAYS[4] == (1, 1)
    assert SINGULAR_CONES[-1] == (0, 1)
    assert P4_RAYS[4] == (-1, -1, -1, -1)
