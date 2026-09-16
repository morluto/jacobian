"""Exact affine-chart and toric-morphism tests for the toric domain.

Every expected value is established by an independent oracle: the dual-of-dual
reconstruction for charts, direct relation replay on the Hilbert basis,
brute-force monoid generation for completeness and indecomposability, and
exact cone-membership replay for morphism assignments. No test imports the
catalog, dispatch, CLI, or MCP product layers.
"""

from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError
from tests.math.geometry.toric._fixtures import (
    A2_CONES,
    A2_RAYS,
    DP6_CONES,
    DP6_RAYS,
    P2_CONES,
    SQUARE_CONES,
    a2_fan,
    dp6_fan,
    fan,
    p2_fan,
    singular_fan,
    square_fan,
)

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.toric import _kernel
from jacobian.math.geometry.toric._kernel import _dual_cone_extreme_rays, _ray_in_cone
from jacobian.math.geometry.toric._models import (
    MAX_TORIC_RAY_COUNT,
    ToricAffineChartRequest,
    ToricAffineChartResult,
    ToricFanPresentation,
    ToricMorphismRequest,
    ToricMorphismResult,
)
from jacobian.math.geometry.toric._tools import (
    AFFINE_CHART_OPERATION,
    MORPHISM_CHECK_OPERATION,
)
from jacobian.math.geometry.toric.operations import (
    check_toric_morphism,
    compute_affine_chart,
)
from jacobian.math.matrices.values import IntegerMatrix


def _matrix(rows: tuple[tuple[int, ...], ...]) -> IntegerMatrix:
    return IntegerMatrix(
        row_count=len(rows),
        column_count=len(rows[0]) if rows else 0,
        entries=rows,
    )


def _dot(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(first * second for first, second in zip(left, right, strict=True))


def _in_dual(point: tuple[int, ...], cone_rays: tuple[tuple[int, ...], ...]) -> bool:
    return all(_dot(ray, point) >= 0 for ray in cone_rays)


def _monoid_points(
    generators: tuple[tuple[int, ...], ...],
    bound: int,
    dimension: int,
) -> set[tuple[int, ...]]:
    """Close the generators under addition inside a coordinate box."""

    closure: set[tuple[int, ...]] = {(0,) * dimension}
    frontier = [(0,) * dimension]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            candidate = tuple(
                current[axis] + generator[axis] for axis in range(dimension)
            )
            if any(abs(value) > bound for value in candidate):
                continue
            if candidate not in closure:
                closure.add(candidate)
                frontier.append(candidate)
    return closure


def _p2_two_cones() -> tuple[tuple[int, ...], ...]:
    return tuple(cone for cone in P2_CONES if len(cone) == 2)


# ---------------------------------------------------------------------------
# Affine charts
# ---------------------------------------------------------------------------


def test_affine_plane_chart_is_the_polynomial_chart() -> None:
    result = compute_affine_chart(a2_fan(), (0, 1))
    assert result.dimension == 2
    assert result.is_smooth is True
    assert result.dual_cone_rays == ((0, 1), (1, 0))
    assert result.hilbert_basis == ((0, 1), (1, 0))
    assert result.relations == ()
    assert [
        (localization.face_cone_id, localization.localizing_character)
        for localization in result.localizations
    ] == [(0, (1, 1)), (1, (0, 1)), (2, (1, 0))]


@pytest.mark.parametrize("cone", _p2_two_cones(), ids=lambda cone: str(cone))
def test_projective_plane_charts_are_smooth_polynomial_rings(
    cone: tuple[int, ...],
) -> None:
    result = compute_affine_chart(p2_fan(), cone)
    assert result.is_smooth is True
    assert len(result.hilbert_basis) == 2
    assert result.relations == ()
    # A smooth two-dimensional chart has a free Hilbert basis and an empty
    # relation lattice.
    assert result.localizations


def test_singular_a1_chart_has_the_quadric_relation() -> None:
    result = compute_affine_chart(singular_fan(), (0, 1))
    assert result.is_smooth is False
    assert result.dual_cone_rays == ((0, 1), (2, -1))
    assert result.hilbert_basis == ((0, 1), (1, 0), (2, -1))
    assert result.relations == ((1, -2, 1),)


def test_relation_lattice_replays_to_zero_on_every_generator() -> None:
    for presentation, cone in ((singular_fan(), (0, 1)), (square_fan(), (0, 1, 2, 3))):
        result = compute_affine_chart(presentation, cone)
        for relation in result.relations:
            assert all(
                sum(
                    relation[index] * generator[axis]
                    for index, generator in enumerate(result.hilbert_basis)
                )
                == 0
                for axis in range(result.lattice_rank)
            )


def test_non_simplicial_square_cone_chart_and_relations() -> None:
    result = compute_affine_chart(square_fan(), (0, 1, 2, 3))
    assert result.dimension == 3
    assert result.is_smooth is False
    assert (1, 0, 0) in result.hilbert_basis
    assert len(result.hilbert_basis) == 5
    assert len(result.relations) == 2
    for relation in result.relations:
        for axis in range(result.lattice_rank):
            assert (
                sum(
                    relation[index] * generator[axis]
                    for index, generator in enumerate(result.hilbert_basis)
                )
                == 0
            )


def test_dual_cone_reconstructs_the_original_cone() -> None:
    for presentation, cone in (
        (a2_fan(), (0, 1)),
        (singular_fan(), (0, 1)),
        (square_fan(), (0, 1, 2, 3)),
    ):
        cone_rays = tuple(presentation.rays[index] for index in cone)
        dual = tuple(
            sorted(_dual_cone_extreme_rays(cone_rays, presentation.lattice_rank))
        )
        assert dual == compute_affine_chart(presentation, cone).dual_cone_rays
        # Double description is an involution on full-dimensional pointed cones.
        reconstructed = tuple(
            sorted(_dual_cone_extreme_rays(dual, presentation.lattice_rank))
        )
        assert reconstructed == tuple(sorted(cone_rays))


def test_hilbert_basis_generates_every_nearby_semigroup_point() -> None:
    presentation = square_fan()
    cone = (0, 1, 2, 3)
    result = compute_affine_chart(presentation, cone)
    cone_rays = tuple(presentation.rays[index] for index in cone)
    bound = max(
        sum(abs(ray[axis]) for ray in result.dual_cone_rays)
        for axis in range(presentation.lattice_rank)
    )
    generated = _monoid_points(result.hilbert_basis, bound, presentation.lattice_rank)
    for point in product(*(range(-bound, bound + 1) for _ in range(3))):
        if point == (0, 0, 0):
            continue
        if _in_dual(point, cone_rays):
            assert point in generated, point


def test_hilbert_basis_elements_are_indecomposable() -> None:
    presentation = singular_fan()
    cone = (0, 1)
    result = compute_affine_chart(presentation, cone)
    cone_rays = tuple(presentation.rays[index] for index in cone)
    bound = max(
        sum(abs(ray[axis]) for ray in result.dual_cone_rays)
        for axis in range(presentation.lattice_rank)
    )
    semigroup = {
        point
        for point in product(*(range(-bound, bound + 1) for _ in range(2)))
        if _in_dual(point, cone_rays)
    }
    for generator in result.hilbert_basis:
        for left in semigroup:
            if left == (0, 0) or left == generator:
                continue
            remainder = tuple(generator[axis] - left[axis] for axis in range(2))
            if remainder == (0, 0):
                continue
            if _in_dual(remainder, cone_rays):
                raise AssertionError(f"{generator} decomposes as {left} + {remainder}")


def test_chart_localizations_are_supporting_characters() -> None:
    presentation = square_fan()
    cone = (0, 1, 2, 3)
    result = compute_affine_chart(presentation, cone)
    cone_rays = tuple(presentation.rays[index] for index in cone)
    for localization in result.localizations:
        face = set(localization.face_ray_indices)
        character = localization.localizing_character
        for ray_index, ray in enumerate(cone_rays):
            pairing = _dot(character, ray)
            if ray_index in face:
                assert pairing == 0
            else:
                assert pairing > 0


def test_del_pezzo_charts_are_smooth() -> None:
    presentation = dp6_fan()
    for cone in DP6_CONES:
        if len(cone) != presentation.lattice_rank:
            continue
        result = compute_affine_chart(presentation, cone)
        assert result.is_smooth is True
        assert len(result.hilbert_basis) == presentation.lattice_rank
        assert result.relations == ()


# ---------------------------------------------------------------------------
# Chart boundaries and adversarial inputs
# ---------------------------------------------------------------------------


def test_rank_one_ray_chart_is_the_affine_line() -> None:
    presentation = ToricFanPresentation(lattice_rank=1, rays=((1,),), cones=((), (0,)))
    result = compute_affine_chart(presentation, (0,))
    assert result.dual_cone_rays == ((1,),)
    assert result.hilbert_basis == ((1,),)
    assert result.relations == ()
    assert [
        (localization.face_cone_id, localization.localizing_character)
        for localization in result.localizations
    ] == [(0, (1,))]


def test_lower_dimensional_cone_is_refused() -> None:
    with pytest.raises(OperationDomainValidationError) as rejection:
        compute_affine_chart(p2_fan(), (0,))
    assert (
        rejection.value.errors()[0]["type"] == "toric.chart_cone_not_full_dimensional"
    )


def test_undeclared_cone_is_refused() -> None:
    with pytest.raises(OperationDomainValidationError) as rejection:
        compute_affine_chart(dp6_fan(), (0, 1))
    assert rejection.value.errors()[0]["type"] == "toric.chart_cone_not_declared"


def test_invalid_fan_is_refused_before_chart_work() -> None:
    cones = tuple(cone for cone in DP6_CONES if cone != (1,))
    with pytest.raises(OperationDomainValidationError) as rejection:
        compute_affine_chart(fan(DP6_RAYS, cones), (0, 3))
    assert rejection.value.errors()[0]["type"] == "toric.fan_not_valid"


def test_out_of_range_cone_label_is_structurally_rejected() -> None:
    with pytest.raises(ValidationError):
        ToricAffineChartRequest(fan=p2_fan(), cone=(0, 5))


def test_over_rank_and_over_ray_charts_are_refused() -> None:
    over_rank = ToricFanPresentation.model_construct(lattice_rank=5, rays=(), cones=())
    with pytest.raises(OperationResourceAdmissionError):
        compute_affine_chart(over_rank, ())
    over_rays = ToricFanPresentation.model_construct(
        lattice_rank=2,
        rays=tuple((index, 1) for index in range(MAX_TORIC_RAY_COUNT + 1)),
        cones=(),
    )
    with pytest.raises(OperationResourceAdmissionError):
        compute_affine_chart(over_rays, ())


def test_unbounded_hilbert_work_is_refused_not_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_kernel, "MAX_TORIC_CHART_BOX", 1)
    with pytest.raises(OperationResourceAdmissionError) as rejection:
        compute_affine_chart(singular_fan(), (0, 1))
    assert rejection.value.errors()[0]["type"] == "toric.resource_budget_exceeded"


# ---------------------------------------------------------------------------
# Toric morphisms
# ---------------------------------------------------------------------------


def test_identity_is_a_toric_morphism_and_assignments_replay() -> None:
    matrix = _matrix(((1, 0), (0, 1)))
    result = check_toric_morphism(p2_fan(), p2_fan(), matrix)
    assert result.is_toric_morphism is True
    assert result.obstruction is None
    assert len(result.assignments) == len(P2_CONES)
    for assignment in result.assignments:
        source_cone = p2_fan().cones[assignment.source_cone_id]
        target_cone = p2_fan().cones[assignment.target_cone_id]
        target_generators = tuple(p2_fan().rays[index] for index in target_cone)
        for ray_index in source_cone:
            assert _ray_in_cone(result.ray_images[ray_index], target_generators)


def test_morphism_obstruction_without_a_common_target_cone() -> None:
    matrix = _matrix(((1, 1), (1, -1)))
    result = check_toric_morphism(p2_fan(), p2_fan(), matrix)
    assert result.is_toric_morphism is False
    obstruction = result.obstruction
    assert obstruction is not None
    assert obstruction.reason == "NO_COMMON_TARGET_CONE"
    assert obstruction.failing_ray_index is None
    # Independently confirm that no single target cone contains both images.
    target_cone_ids = set(range(len(P2_CONES)))
    for candidates in obstruction.candidate_target_cone_ids:
        target_cone_ids &= set(candidates)
    assert target_cone_ids == set()


def test_morphism_obstruction_with_a_ray_outside_all_target_cones() -> None:
    matrix = _matrix(((1, 0), (0, 1)))
    result = check_toric_morphism(p2_fan(), a2_fan(), matrix)
    assert result.is_toric_morphism is False
    obstruction = result.obstruction
    assert obstruction is not None
    assert obstruction.reason == "RAY_IMAGE_OUTSIDE_ALL_TARGET_CONES"
    assert obstruction.failing_ray_index == 2
    assert obstruction.failing_image == (-1, -1)
    target_rays = tuple(a2_fan().rays[index] for index in range(len(A2_RAYS)))
    assert not _ray_in_cone((-1, -1), target_rays)


def test_morphism_matrix_rank_mismatch_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ToricMorphismRequest(
            source=p2_fan(),
            target=p2_fan(),
            matrix=_matrix(((1, 0, 0), (0, 1, 0))),
        )
    mismatched = IntegerMatrix.model_construct(
        row_count=2, column_count=2, entries=((1, 0), (0, 1), (1, 1))
    )
    with pytest.raises(OperationDomainValidationError) as rejection:
        check_toric_morphism(p2_fan(), p2_fan(), mismatched)
    assert rejection.value.errors()[0]["type"] == "toric.morphism_matrix_shape"


def test_invalid_source_fan_is_refused_before_morphism_work() -> None:
    invalid = fan(A2_RAYS, ((), (0,), (0, 1)))
    with pytest.raises(OperationDomainValidationError) as rejection:
        check_toric_morphism(invalid, a2_fan(), _matrix(((1, 0), (0, 1))))
    assert rejection.value.errors()[0]["type"] == "toric.fan_not_valid"


# ---------------------------------------------------------------------------
# Serialization and catalog parity
# ---------------------------------------------------------------------------


def test_chart_round_trips_through_strict_json() -> None:
    result = compute_affine_chart(singular_fan(), (0, 1))
    replayed = ToricAffineChartResult.model_validate_json(result.model_dump_json())
    assert replayed == result


def test_morphism_round_trips_through_strict_json() -> None:
    qualifying = check_toric_morphism(p2_fan(), p2_fan(), _matrix(((1, 0), (0, 1))))
    assert (
        ToricMorphismResult.model_validate_json(qualifying.model_dump_json())
        == qualifying
    )
    obstructed = check_toric_morphism(p2_fan(), p2_fan(), _matrix(((1, 1), (1, -1))))
    replayed = ToricMorphismResult.model_validate_json(obstructed.model_dump_json())
    assert replayed == obstructed
    assert replayed.obstruction is not None


def test_catalog_and_native_chart_paths_agree() -> None:
    payload = {
        "fan": {
            "lattice_rank": 2,
            "rays": [["1", "0"], ["0", "1"]],
            "cones": [[], [0], [1], [0, 1]],
        },
        "cone": [0, 1],
    }
    request = ToricAffineChartRequest.model_validate_json(
        encode_strict_json(payload), strict=True
    )
    catalog_result = AFFINE_CHART_OPERATION.run(request)
    native_result = compute_affine_chart(a2_fan(), (0, 1))
    assert catalog_result == native_result
    assert catalog_result.fan == a2_fan()


def test_catalog_and_native_morphism_paths_agree() -> None:
    payload = {
        "source": {
            "lattice_rank": 2,
            "rays": [["1", "0"], ["0", "1"], ["-1", "-1"]],
            "cones": [[], [0], [1], [2], [0, 1], [0, 2], [1, 2]],
        },
        "target": {
            "lattice_rank": 2,
            "rays": [["1", "0"], ["0", "1"], ["-1", "-1"]],
            "cones": [[], [0], [1], [2], [0, 1], [0, 2], [1, 2]],
        },
        "matrix": {
            "row_count": 2,
            "column_count": 2,
            "entries": [["1", "0"], ["0", "1"]],
        },
    }
    request = ToricMorphismRequest.model_validate_json(
        encode_strict_json(payload), strict=True
    )
    catalog_result = MORPHISM_CHECK_OPERATION.run(request)
    native_result = check_toric_morphism(p2_fan(), p2_fan(), _matrix(((1, 0), (0, 1))))
    assert catalog_result == native_result
    assert catalog_result.is_toric_morphism is True


def test_known_counts_and_fixtures_are_retained() -> None:
    assert len(A2_CONES) == 4
    assert len(P2_CONES) == 7
    assert len(DP6_CONES) == 13
    assert len(SQUARE_CONES) == 10
