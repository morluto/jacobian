from __future__ import annotations

import pytest
from pydantic import ValidationError

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
