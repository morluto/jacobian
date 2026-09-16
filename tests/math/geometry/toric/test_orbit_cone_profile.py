"""Known-answer tests for the orbit-cone correspondence profile."""

from __future__ import annotations

import pytest
from tests.math.geometry.toric._fixtures import (
    A2_CONES,
    DP6_CONES,
    P2_CONES,
    P4_CONES,
    SQUARE_RAYS,
    a2_fan,
    dp6_fan,
    fan,
    p2_fan,
    p4_fan,
    singular_fan,
    square_fan,
)

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import parse_operation_input
from jacobian.math.geometry.toric._models import (
    OrbitConeProfileRequest,
    OrbitConeProfileResult,
)
from jacobian.math.geometry.toric._tools import ORBIT_CONE_PROFILE_OPERATION
from jacobian.math.geometry.toric.operations import compute_orbit_cone_profile


def _rows(result: OrbitConeProfileResult) -> dict[int, tuple[int, int, bool, bool]]:
    return {
        row.cone_id: (
            row.dimension,
            row.orbit_dimension,
            row.is_simplicial,
            row.is_smooth,
        )
        for row in result.cones
    }


def test_affine_plane_profile_matches_the_orbit_cone_correspondence() -> None:
    result = compute_orbit_cone_profile(a2_fan())
    assert len(result.cones) == 4
    rows = _rows(result)
    assert rows[0] == (0, 2, True, True)  # zero cone: dense torus orbit
    assert rows[1] == (1, 1, True, True)
    assert rows[2] == (1, 1, True, True)
    assert rows[3] == (2, 0, True, True)
    assert result.lattice_rank == 2
    assert tuple(
        (relation.tau_cone_id, relation.sigma_cone_id)
        for relation in result.face_relations
    ) == (
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 1), (1, 3),
        (2, 2), (2, 3),
        (3, 3),
    )


def test_projective_plane_has_seven_cones_and_orbit_dimensions() -> None:
    result = compute_orbit_cone_profile(p2_fan())
    # Sigma_{P^2} = {0} + 3 rays + 3 two-dimensional cones = 7 cones, hence
    # torus orbits of dimensions 2, 1, 1, 1, 0, 0, 0.
    assert len(result.cones) == 7
    orbit_dimensions = sorted(
        (row.dimension, row.orbit_dimension) for row in result.cones
    )
    assert orbit_dimensions == [(0, 2), (1, 1), (1, 1), (1, 1), (2, 0), (2, 0), (2, 0)]
    assert all(row.is_smooth for row in result.cones)
    # Face relations: zero cone under all 7; each ray under itself and its two
    # two-dimensional cones; each surface cone under itself: 7 + 3*3 + 3 = 19.
    assert len(result.face_relations) == 19


def test_projective_plane_face_relations_are_exactly_subset_inclusion() -> None:
    presentation = p2_fan()
    result = compute_orbit_cone_profile(presentation)
    cones = tuple(sorted(cone) for cone in P2_CONES)
    expected = tuple(
        (tau, sigma)
        for tau in range(len(cones))
        for sigma in range(len(cones))
        if tau <= sigma and set(cones[tau]) <= set(cones[sigma])
    )
    assert tuple(
        (relation.tau_cone_id, relation.sigma_cone_id)
        for relation in result.face_relations
    ) == expected


def test_degree_six_del_pezzo_profile_is_smooth_with_six_two_cones() -> None:
    result = compute_orbit_cone_profile(dp6_fan())
    assert len(result.cones) == 13
    assert all(row.is_smooth for row in result.cones)
    assert sum(1 for row in result.cones if row.dimension == 1) == 6
    assert sum(1 for row in result.cones if row.dimension == 2) == 6
    assert len(DP6_CONES) == 13


def test_projective_four_space_fan_is_smooth_in_rank_four() -> None:
    result = compute_orbit_cone_profile(p4_fan())
    assert len(result.cones) == 31
    assert result.lattice_rank == 4
    assert all(row.is_smooth for row in result.cones)
    assert sum(1 for row in result.cones if row.dimension == 4) == 5
    zero_cone = result.cones[0]
    assert (zero_cone.dimension, zero_cone.orbit_dimension) == (0, 4)


def test_a_two_quotient_cone_is_simplicial_but_not_smooth() -> None:
    result = compute_orbit_cone_profile(singular_fan())
    top = result.cones[3]
    assert top.ray_indices == (0, 1)
    assert top.dimension == 2
    assert top.orbit_dimension == 0
    assert top.is_simplicial is True
    # det((1,0),(1,2)) = 2, so the Smith form has invariant factor 2.
    assert top.is_smooth is False
    assert all(row.is_smooth for row in result.cones[:3])


def test_square_pyramid_cone_is_not_simplicial() -> None:
    result = compute_orbit_cone_profile(square_fan())
    rows = {row.cone_id: row for row in result.cones}
    full = rows[9]
    assert full.ray_indices == (0, 1, 2, 3)
    assert full.dimension == 3
    assert full.orbit_dimension == 0
    assert full.is_simplicial is False
    assert full.is_smooth is False
    assert len(result.cones) == 10
    assert len(SQUARE_RAYS) == 4


def test_zero_cone_only_fan_profile() -> None:
    result = compute_orbit_cone_profile(fan((), ((),), rank=3))
    assert len(result.cones) == 1
    row = result.cones[0]
    assert (row.dimension, row.orbit_dimension) == (0, 3)
    assert row.is_smooth is True
    assert result.face_relations[0].tau_cone_id == 0
    assert result.ray_incidence == ()


def test_single_ray_fan_profile_and_incidence() -> None:
    result = compute_orbit_cone_profile(fan(((1, 0),), ((), (0,)), rank=2))
    assert [
        (row.dimension, row.orbit_dimension) for row in result.cones
    ] == [(0, 2), (1, 1)]
    assert tuple(
        (relation.tau_cone_id, relation.sigma_cone_id)
        for relation in result.face_relations
    ) == ((0, 0), (0, 1), (1, 1))
    assert result.ray_incidence[0].cone_ids == (1,)


def test_ray_incidence_lists_every_cone_containing_each_ray() -> None:
    result = compute_orbit_cone_profile(dp6_fan())
    presentation_cones = tuple(sorted(cone) for cone in DP6_CONES)
    for ray_index in range(len(result.fan.rays)):
        expected = tuple(
            cone_id
            for cone_id, cone in enumerate(presentation_cones)
            if ray_index in cone
        )
        assert result.ray_incidence[ray_index].cone_ids == expected


def test_invalid_fan_is_rejected_with_the_owner_code() -> None:
    from tests.math.geometry.toric._fixtures import DP6_RAYS

    cones = tuple(cone for cone in DP6_CONES if cone != (1,))
    with pytest.raises(OperationDomainValidationError) as rejection:
        compute_orbit_cone_profile(fan(DP6_RAYS, cones))
    assert rejection.value.errors()[0]["type"] == "toric.fan_not_valid"


def test_catalog_and_native_profiles_agree_and_round_trip() -> None:
    payload = {
        "fan": {
            "lattice_rank": 2,
            "rays": [["1", "0"], ["0", "1"]],
            "cones": [[], [0], [1], [0, 1]],
        }
    }
    request = parse_operation_input(OrbitConeProfileRequest, payload)
    catalog_result = ORBIT_CONE_PROFILE_OPERATION.run(request)
    native_result = compute_orbit_cone_profile(a2_fan())
    assert catalog_result == native_result
    replayed = OrbitConeProfileResult.model_validate_json(
        native_result.model_dump_json()
    )
    assert replayed == native_result


def test_profile_recognizes_the_serialized_claim_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.geometry.toric import operations

    calls = 0
    original = operations.recognize_fan

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(operations, "recognize_fan", counted)
    result = compute_orbit_cone_profile(p2_fan())
    assert calls == 1
    assert len(result.cones) == 7
    assert len(P2_CONES) == 7
    assert len(A2_CONES) == 4
    assert len(P4_CONES) == 31
