from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.codes.general import minimum_distance
from jacobian.math.combinatorics.codes.general._models import (
    CoveringRadiusRequest,
    LinearCodeRequest,
)
from jacobian.math.combinatorics.codes.general._tools import (
    _covering_radius,
    _minimum_distance,
    _weight_distribution,
)


def _assert_validation_error_code(factory: Callable[[], object], code: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        factory()
    assert exc_info.value.errors()[0]["type"] == code


def _assert_operation_error(factory: Callable[[], object], code: str) -> None:
    with pytest.raises(OperationDomainValidationError) as exc_info:
        factory()
    assert exc_info.value.errors()[0]["type"] == code


def test_prime_field_code_enumeration_uses_the_declared_matrix() -> None:
    request = LinearCodeRequest(field_order=2, generator_matrix=((1, 1),))

    assert _minimum_distance(request).minimum_distance == 2
    assert _weight_distribution(request).weights == ((0, 1), (2, 1))


def test_code_weight_distribution_counts_distinct_words_for_dependent_rows() -> None:
    request = LinearCodeRequest(field_order=2, generator_matrix=((1,), (1,)))

    assert _weight_distribution(request).weights == ((0, 1), (1, 1))


def test_code_contract_rejects_nonprime_fields_and_unbounded_enumeration() -> None:
    _assert_operation_error(
        lambda: _minimum_distance(
            LinearCodeRequest(field_order=4, generator_matrix=((1,),))
        ),
        "code_theory.field_order_not_prime",
    )
    _assert_operation_error(
        lambda: _minimum_distance(
            LinearCodeRequest(field_order=251, generator_matrix=((1,),) * 3)
        ),
        "code_theory.enumeration_work_exceeded",
    )


def test_native_code_api_enforces_the_prime_field_contract() -> None:
    assert minimum_distance(((1, 1),), 2) == 2

    _assert_operation_error(
        lambda: minimum_distance(((1,),), 4), "code_theory.field_order_not_prime"
    )


def test_zero_code_uses_length_convention_for_minimum_distance() -> None:
    assert minimum_distance(((0, 0, 0, 0),), 2) == 4


def test_native_code_api_rejects_empty_generator_matrix_structurally() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        minimum_distance((), 2)
    assert error.value.errors()[0]["type"] == "code_theory.generator_matrix_empty"


def test_native_code_api_rejects_unequal_rows_semantically() -> None:
    _assert_operation_error(
        lambda: minimum_distance(((1, 0), (1,)), 2),
        "code_theory.generator_rows_unequal",
    )


def test_binary_repetition_code_length_three_has_covering_radius_one() -> None:
    request = CoveringRadiusRequest(
        field_order=2,
        generator_matrix=((1, 1, 1),),
    )

    result = _covering_radius(request)

    assert result.covering_radius == 1


def test_binary_repetition_code_length_four_has_covering_radius_two() -> None:
    request = CoveringRadiusRequest(
        field_order=2,
        generator_matrix=((1, 1, 1, 1),),
    )

    assert _covering_radius(request).covering_radius == 2


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

    assert _covering_radius(request).covering_radius == 1


def test_ternary_repetition_code_has_covering_radius_two() -> None:
    request = CoveringRadiusRequest(
        field_order=3,
        generator_matrix=((1, 1, 1),),
    )

    assert _covering_radius(request).covering_radius == 2


def test_dependent_generator_rows_use_rank_not_row_count() -> None:
    request = CoveringRadiusRequest(
        field_order=2,
        generator_matrix=((1, 1, 1), (1, 1, 1)),
    )

    assert _covering_radius(request).covering_radius == 1


def test_full_space_code_has_covering_radius_zero() -> None:
    request = CoveringRadiusRequest(
        field_order=2,
        generator_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )

    assert _covering_radius(request).covering_radius == 0


def test_zero_code_has_covering_radius_equal_to_length() -> None:
    request = CoveringRadiusRequest(
        field_order=2,
        generator_matrix=((0, 0, 0),),
    )

    assert _covering_radius(request).covering_radius == 3


def test_covering_radius_contract_rejects_dependent_row_state_space_hole() -> None:
    repeated_row = (1, 0, 0, 0, 0, 0, 0, 0)

    _assert_operation_error(
        lambda: _covering_radius(
            CoveringRadiusRequest(
                field_order=251,
                generator_matrix=(repeated_row,) * 8,
            )
        ),
        "code_theory.syndrome_state_bound_exceeded",
    )


def test_covering_radius_contract_rejects_excessive_transition_work() -> None:
    generator_matrix = tuple(
        tuple(1 if column == row else 0 for column in range(18)) for row in range(8)
    )

    _assert_operation_error(
        lambda: _covering_radius(
            CoveringRadiusRequest(
                field_order=3,
                generator_matrix=generator_matrix,
            )
        ),
        "code_theory.syndrome_transition_bound_exceeded",
    )


def test_covering_radius_contract_rejects_nonprime_field() -> None:
    _assert_operation_error(
        lambda: _covering_radius(
            CoveringRadiusRequest(
                field_order=4,
                generator_matrix=((1, 1),),
            )
        ),
        "code_theory.field_order_not_prime",
    )


# ---------------------------------------------------------------------------
# Source-bound replay regressions (#2309)
# ---------------------------------------------------------------------------


def _linear_request() -> LinearCodeRequest:
    return LinearCodeRequest(field_order=3, generator_matrix=((1, 0, 1), (0, 1, 1)))


def test_results_retain_source_and_round_trip() -> None:
    from jacobian.math.combinatorics.codes.general._models import (
        CoveringRadiusResult,
        MinimumDistanceResult,
        WeightDistributionResult,
    )

    request = _linear_request()
    dist = _minimum_distance(request)
    assert dist.request == request
    assert dist.minimum_distance == 2
    assert MinimumDistanceResult.model_validate(dist.model_dump()) == dist

    profile = _weight_distribution(request)
    assert profile.request == request
    # q^rank = 9 distinct codewords over GF(3) with rank-2 generator rows.
    assert sum(count for _weight, count in profile.weights) == 9
    assert WeightDistributionResult.model_validate(profile.model_dump()) == profile

    covering = CoveringRadiusRequest(
        field_order=2, generator_matrix=((1, 0, 1), (0, 1, 1))
    )
    radius = _covering_radius(covering)
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


def test_structural_models_accept_bounded_claims() -> None:
    from jacobian.math.combinatorics.codes.general._models import (
        MinimumDistanceResult,
        WeightDistributionResult,
    )

    request = _linear_request()
    base = {"request": request.model_dump()}

    wrong_distance = MinimumDistanceResult.model_validate(
        dict(base, minimum_distance=1)
    )
    assert wrong_distance.minimum_distance == 1

    foreign_code = LinearCodeRequest(
        field_order=2, generator_matrix=((1, 0), (0, 1), (1, 1))
    )
    forged_source = MinimumDistanceResult.model_validate(
        dict(base, request=foreign_code.model_dump(), minimum_distance=2)
    )
    assert forged_source.request != request

    unsorted_profile = dict(base, weights=((1, 6), (0, 1), (2, 2)))
    _assert_validation_error_code(
        lambda: WeightDistributionResult.model_validate(unsorted_profile),
        "code_theory.weights_not_strictly_ascending",
    )

    bad_total = WeightDistributionResult.model_validate(
        dict(base, weights=((0, 1), (2, 2)))
    )
    assert bad_total.weights == ((0, 1), (2, 2))

    forged_profile = WeightDistributionResult.model_validate(
        dict(base, weights=((0, 1), (1, 5), (2, 3)))
    )
    assert forged_profile.weights == ((0, 1), (1, 5), (2, 3))


def test_zero_code_result_preserves_the_documented_length_convention() -> None:
    from jacobian.math.combinatorics.codes.general._models import MinimumDistanceResult

    request = LinearCodeRequest(field_order=2, generator_matrix=((0, 0, 0),))
    result = _minimum_distance(request)

    assert result.minimum_distance == 3
    assert MinimumDistanceResult.model_validate(result.model_dump()) == result
    description = MinimumDistanceResult.model_json_schema()["description"]
    assert "empty-code convention" in description


def test_dependent_generator_rows_rank_cardinality() -> None:
    """Dependent rows deduplicate: cardinality is q^rank, not q^rows."""

    request = LinearCodeRequest(field_order=2, generator_matrix=((1, 1), (1, 1)))
    profile = _weight_distribution(request)
    assert sum(count for _weight, count in profile.weights) == 2


def test_enumeration_budget_charges_the_selected_kernel_path() -> None:
    """Admission charges the exact enumeration selected by the operation."""

    from jacobian.math.combinatorics.codes.general._models import (
        EXACT_ENUMERATION_PASSES,
        MAX_EXACT_CODEWORD_EVALUATIONS,
    )

    assert EXACT_ENUMERATION_PASSES == 1

    per_pass = MAX_EXACT_CODEWORD_EVALUATIONS // EXACT_ENUMERATION_PASSES
    boundary = LinearCodeRequest(field_order=251, generator_matrix=((1,), (0,)))
    tuples_per_pass = boundary.field_order ** len(boundary.generator_matrix)
    assert tuples_per_pass <= per_pass
    assert _minimum_distance(boundary).minimum_distance == 1
    assert _weight_distribution(boundary).weights == ((0, 1), (1, 250))

    _assert_operation_error(
        lambda: _minimum_distance(
            LinearCodeRequest(field_order=251, generator_matrix=((1,),) * 3)
        ),
        "code_theory.enumeration_work_exceeded",
    )


def test_covering_radius_budget_charges_the_selected_bfs_path() -> None:
    """Syndrome-graph admission charges the operation's one BFS pass."""

    from jacobian.math.combinatorics.codes.general._models import (
        MAX_COVERING_RADIUS_STATES_PER_PASS,
        MAX_COVERING_RADIUS_TRANSITIONS,
        SYNDROME_BFS_PASSES,
    )

    assert SYNDROME_BFS_PASSES == 1

    identity = tuple(
        tuple(1 if column == row else 0 for column in range(8)) for row in range(8)
    )
    boundary_matrix = tuple(row + (1,) * 16 for row in identity)
    boundary = CoveringRadiusRequest(field_order=2, generator_matrix=boundary_matrix)
    width = len(boundary.generator_matrix[0])
    states = 2 ** (width - 8)
    assert states == MAX_COVERING_RADIUS_STATES_PER_PASS
    assert states * 24 * SYNDROME_BFS_PASSES <= MAX_COVERING_RADIUS_TRANSITIONS

    _assert_operation_error(
        lambda: _covering_radius(
            CoveringRadiusRequest(
                field_order=2,
                generator_matrix=tuple(row + (1,) * 17 for row in identity),
            )
        ),
        "code_theory.syndrome_state_bound_exceeded",
    )
