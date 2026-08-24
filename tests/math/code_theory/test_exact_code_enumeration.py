from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.code_theory import minimum_distance
from jacobian.math.code_theory._models import (
    CoveringRadiusRequest,
    LinearCodeRequest,
)
from jacobian.math.code_theory._operations import (
    compute_covering_radius,
    compute_min_distance,
    compute_weight_dist,
)


def test_prime_field_code_enumeration_uses_the_declared_matrix() -> None:
    request = LinearCodeRequest(field_order=2, generator_matrix=((1, 1),))

    assert compute_min_distance(request).minimum_distance == 2
    assert compute_weight_dist(request).weights == ((0, 1), (2, 1))


def test_code_weight_distribution_counts_distinct_words_for_dependent_rows() -> None:
    request = LinearCodeRequest(field_order=2, generator_matrix=((1,), (1,)))

    assert compute_weight_dist(request).weights == ((0, 1), (1, 1))


def test_code_contract_rejects_nonprime_fields_and_unbounded_enumeration() -> None:
    with pytest.raises(ValidationError, match="prime"):
        LinearCodeRequest(field_order=4, generator_matrix=((1,),))
    with pytest.raises(ValidationError, match="enumeration"):
        LinearCodeRequest(field_order=251, generator_matrix=((1,), (1,), (1,)))


def test_native_code_api_enforces_the_prime_field_contract() -> None:
    assert minimum_distance(((1, 1),), 2) == 2

    with pytest.raises(ValidationError, match="prime"):
        minimum_distance(((1,),), 4)


def test_zero_code_uses_length_convention_for_minimum_distance() -> None:
    assert minimum_distance(((0, 0, 0, 0),), 2) == 4


@pytest.mark.parametrize("generator_matrix", [(), ((1, 0), (1,))])
def test_native_code_api_rejects_invalid_generator_shapes(
    generator_matrix: tuple[tuple[int, ...], ...],
) -> None:
    with pytest.raises(ValidationError, match=r"at least 1|equal length"):
        minimum_distance(generator_matrix, 2)


def test_binary_repetition_code_length_three_has_covering_radius_one() -> None:
    request = CoveringRadiusRequest(
        field_order=2,
        generator_matrix=((1, 1, 1),),
    )

    result = compute_covering_radius(request)

    assert result.covering_radius == 1
    assert result.method == "SYNDROME_BFS"


def test_binary_repetition_code_length_four_has_covering_radius_two() -> None:
    request = CoveringRadiusRequest(
        field_order=2,
        generator_matrix=((1, 1, 1, 1),),
    )

    assert compute_covering_radius(request).covering_radius == 2


def test_binary_hamming_code_has_covering_radius_one() -> None:
    request = CoveringRadiusRequest(
        field_order=2,
        generator_matrix=(
            (1, 0, 0, 0, 0, 1, 1),
            (0, 1, 0, 0, 1, 0, 1),
            (0, 0, 1, 0, 1, 1, 0),
            (0, 0, 0, 1, 1, 1, 1),
        ),
    )

    assert compute_covering_radius(request).covering_radius == 1


def test_ternary_repetition_code_has_covering_radius_two() -> None:
    request = CoveringRadiusRequest(
        field_order=3,
        generator_matrix=((1, 1, 1),),
    )

    assert compute_covering_radius(request).covering_radius == 2


def test_dependent_generator_rows_use_rank_not_row_count() -> None:
    request = CoveringRadiusRequest(
        field_order=2,
        generator_matrix=((1, 1, 1), (1, 1, 1)),
    )

    assert compute_covering_radius(request).covering_radius == 1


def test_full_space_code_has_covering_radius_zero() -> None:
    request = CoveringRadiusRequest(
        field_order=2,
        generator_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )

    assert compute_covering_radius(request).covering_radius == 0


def test_zero_code_has_covering_radius_equal_to_length() -> None:
    request = CoveringRadiusRequest(
        field_order=2,
        generator_matrix=((0, 0, 0),),
    )

    assert compute_covering_radius(request).covering_radius == 3


def test_covering_radius_contract_rejects_dependent_row_state_space_hole() -> None:
    repeated_row = (1, 0, 0, 0, 0, 0, 0, 0)

    with pytest.raises(ValidationError, match="syndrome space"):
        CoveringRadiusRequest(
            field_order=251,
            generator_matrix=(repeated_row,) * 8,
        )


def test_covering_radius_contract_rejects_excessive_transition_work() -> None:
    generator_matrix = tuple(
        tuple(1 if column == row else 0 for column in range(18)) for row in range(8)
    )

    with pytest.raises(ValidationError, match="transition"):
        CoveringRadiusRequest(
            field_order=3,
            generator_matrix=generator_matrix,
        )


def test_covering_radius_contract_rejects_nonprime_field() -> None:
    with pytest.raises(ValidationError, match="prime"):
        CoveringRadiusRequest(
            field_order=4,
            generator_matrix=((1, 1),),
        )


# ---------------------------------------------------------------------------
# Source-bound replay regressions (#2309)
# ---------------------------------------------------------------------------


def _linear_request() -> LinearCodeRequest:
    return LinearCodeRequest(field_order=3, generator_matrix=((1, 0, 1), (0, 1, 1)))


def test_results_retain_source_and_replay() -> None:
    from jacobian.math.code_theory._models import (
        CoveringRadiusResult,
        MinimumDistanceResult,
        WeightDistributionResult,
    )

    request = _linear_request()
    dist = compute_min_distance(request)
    assert dist.request == request
    assert dist.minimum_distance == 2
    assert MinimumDistanceResult.model_validate(dist.model_dump()) == dist

    profile = compute_weight_dist(request)
    assert profile.request == request
    # q^rank = 9 distinct codewords over GF(3) with rank-2 generator rows.
    assert sum(count for _weight, count in profile.weights) == 9
    assert WeightDistributionResult.model_validate(profile.model_dump()) == profile

    covering = CoveringRadiusRequest(
        field_order=2, generator_matrix=((1, 0, 1), (0, 1, 1))
    )
    radius = compute_covering_radius(covering)
    assert radius.request == covering
    assert CoveringRadiusResult.model_validate(radius.model_dump()) == radius

    with pytest.raises(ValidationError):
        MinimumDistanceResult(
            request=request,
            minimum_distance=9999,
        )
    with pytest.raises(ValidationError):
        WeightDistributionResult(
            request=request,
            weights=((99, 777), (-4, 123)),
        )
    with pytest.raises(ValidationError):
        CoveringRadiusResult(request=covering, covering_radius=200)


def test_forged_profiles_are_rejected() -> None:
    from jacobian.math.code_theory._models import (
        MinimumDistanceResult,
        WeightDistributionResult,
    )

    request = _linear_request()
    base = {
        "request": request.model_dump(),
        "method": "EXACT_ENUMERATION",
    }

    wrong_distance = dict(base, minimum_distance=1)
    with pytest.raises(ValidationError, match="exact enumeration"):
        MinimumDistanceResult.model_validate(wrong_distance)

    foreign_code = LinearCodeRequest(
        field_order=2, generator_matrix=((1, 0), (0, 1), (1, 1))
    )
    forged_source = dict(base, request=foreign_code.model_dump(), minimum_distance=2)
    with pytest.raises(ValidationError, match="retained source code"):
        MinimumDistanceResult.model_validate(forged_source)

    unsorted_profile = dict(base, weights=((1, 6), (0, 1), (2, 2)))
    with pytest.raises(ValidationError, match="ascending"):
        WeightDistributionResult.model_validate(unsorted_profile)

    bad_total = dict(base, weights=((0, 1), (2, 2)))
    with pytest.raises(ValidationError, match="distinct generated codeword"):
        WeightDistributionResult.model_validate(bad_total)

    forged_profile = dict(base, weights=((0, 1), (1, 5), (2, 3)))
    with pytest.raises(ValidationError, match="exact enumeration"):
        WeightDistributionResult.model_validate(forged_profile)


def test_source_bound_result_versions_track_wire_shape() -> None:
    from jacobian.math.code_theory._tools import TOOLS

    versions = {tool.operation_id: tool.version for tool in TOOLS}

    assert versions["code.minimum_distance.compute"] == "2"
    assert versions["code.weight_distribution.compute"] == "2"
    assert versions["code.covering_radius.compute"] == "2"


def test_zero_code_result_replays_the_documented_length_convention() -> None:
    from jacobian.math.code_theory._models import MinimumDistanceResult

    request = LinearCodeRequest(field_order=2, generator_matrix=((0, 0, 0),))
    result = compute_min_distance(request)

    assert result.minimum_distance == 3
    assert MinimumDistanceResult.model_validate(result.model_dump()) == result
    description = MinimumDistanceResult.model_json_schema()["description"]
    assert "empty-code convention" in description


def test_forged_zero_code_distance_is_rejected() -> None:
    from jacobian.math.code_theory._models import MinimumDistanceResult

    request = LinearCodeRequest(field_order=2, generator_matrix=((0, 0, 0),))
    forged = {
        "request": request.model_dump(),
        "method": "EXACT_ENUMERATION",
        "minimum_distance": 2,
    }

    with pytest.raises(ValidationError, match="exact enumeration"):
        MinimumDistanceResult.model_validate(forged)


def test_dependent_generator_rows_rank_cardinality() -> None:
    """Dependent rows deduplicate: cardinality is q^rank, not q^rows."""

    request = LinearCodeRequest(field_order=2, generator_matrix=((1, 1), (1, 1)))
    profile = compute_weight_dist(request)
    assert sum(count for _weight, count in profile.weights) == 2


def test_enumeration_budget_charges_kernel_and_replay_passes() -> None:
    """Admission charges both exhaustive passes of a source-bound call.

    The kernel pass in ``compute_*`` plus the retained-source replay in
    result validation must jointly fit ``MAX_EXACT_CODEWORD_EVALUATIONS``,
    so the per-pass envelope stays at half that total.
    """

    from jacobian.math.code_theory._models import (
        EXACT_ENUMERATION_PASSES,
        MAX_EXACT_CODEWORD_EVALUATIONS,
    )

    assert EXACT_ENUMERATION_PASSES == 2

    per_pass = MAX_EXACT_CODEWORD_EVALUATIONS // EXACT_ENUMERATION_PASSES
    boundary = LinearCodeRequest(field_order=251, generator_matrix=((1,), (0,)))
    tuples_per_pass = boundary.field_order ** len(boundary.generator_matrix)
    assert tuples_per_pass <= per_pass
    assert compute_min_distance(boundary).minimum_distance == 1
    assert compute_weight_dist(boundary).weights == ((0, 1), (1, 250))

    with pytest.raises(ValidationError, match="enumeration"):
        LinearCodeRequest(field_order=41, generator_matrix=((1,),) * 3)


def test_covering_radius_budget_charges_bfs_and_replay_passes() -> None:
    """Syndrome-graph admission charges both BFS passes' transitions.

    A full-rank width-24 binary code keeps 65,536 syndrome states and
    1,572,864 transitions per pass, so both passes fit the transition
    total; one more column doubles the state space past its per-pass cap.
    """

    from jacobian.math.code_theory._models import (
        MAX_COVERING_RADIUS_STATES_PER_PASS,
        MAX_COVERING_RADIUS_TRANSITIONS,
        SYNDROME_BFS_PASSES,
    )

    assert SYNDROME_BFS_PASSES == 2

    identity = tuple(
        tuple(1 if column == row else 0 for column in range(8)) for row in range(8)
    )
    boundary_matrix = tuple(row + (1,) * 16 for row in identity)
    boundary = CoveringRadiusRequest(field_order=2, generator_matrix=boundary_matrix)
    width = len(boundary.generator_matrix[0])
    states = 2 ** (width - 8)
    assert states == MAX_COVERING_RADIUS_STATES_PER_PASS
    assert states * 24 * SYNDROME_BFS_PASSES <= MAX_COVERING_RADIUS_TRANSITIONS

    with pytest.raises(ValidationError, match="syndrome space"):
        CoveringRadiusRequest(
            field_order=2,
            generator_matrix=tuple(row + (1,) * 17 for row in identity),
        )
