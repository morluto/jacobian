"""Exact tests for principal toric character divisors div(chi^m)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from tests.math.geometry.toric._fixtures import (
    DP6_CONES,
    P2_RAYS,
    a2_fan,
    character,
    dp6_fan,
    fan,
    p2_fan,
    p4_fan,
    square_fan,
)

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import parse_operation_input
from jacobian.math.geometry.toric._models import (
    CharacterDivisorRequest,
    CharacterDivisorResult,
    ToricFanPresentation,
)
from jacobian.math.geometry.toric._tools import CHARACTER_DIVISOR_OPERATION
from jacobian.math.geometry.toric.operations import compute_character_divisor


def test_zero_character_gives_the_zero_divisor() -> None:
    result = compute_character_divisor(p2_fan(), character(0, 0))
    assert result.coefficients == (0, 0, 0)
    assert result.support_ray_indices == ()
    assert result.is_principal is True


def test_projective_plane_pairings_are_exact_with_zero_retained() -> None:
    result = compute_character_divisor(p2_fan(), character(1, 0))
    # <(1,0),(1,0)>=1, <(1,0),(0,1)>=0, <(1,0),(-1,-1)>=-1
    assert result.coefficients == (1, 0, -1)
    assert result.support_ray_indices == (0, 2)


def test_negative_and_mixed_sign_pairings() -> None:
    result = compute_character_divisor(dp6_fan(), character(3, -5))
    expected = tuple(3 * v0 - 5 * v1 for v0, v1 in result.fan.rays)
    assert result.coefficients == expected
    assert result.coefficients == (3, -5, -3, 5, -2, 2)
    assert result.support_ray_indices == (0, 1, 2, 3, 4, 5)


def test_divisor_is_additive_in_the_character() -> None:
    first = compute_character_divisor(p2_fan(), character(2, -1))
    second = compute_character_divisor(p2_fan(), character(-3, 4))
    total = compute_character_divisor(p2_fan(), character(-1, 3))
    assert tuple(a + b for a, b in zip(first.coefficients, second.coefficients,
                                       strict=True)) == total.coefficients
    assert tuple(
        index for index, value in enumerate(total.coefficients) if value != 0
    ) == total.support_ray_indices


def test_divisor_is_negated_by_the_inverse_character() -> None:
    forward = compute_character_divisor(dp6_fan(), character(1, 2))
    inverse = compute_character_divisor(dp6_fan(), character(-1, -2))
    assert tuple(-value for value in forward.coefficients) == inverse.coefficients


@pytest.mark.parametrize(
    "presentation",
    [a2_fan(), p2_fan(), dp6_fan(), square_fan(), p4_fan()],
)
def test_coefficients_pair_the_character_with_every_ray(
    presentation: ToricFanPresentation,
) -> None:
    m = tuple(
        1 if index % 2 == 0 else -2 for index in range(presentation.lattice_rank)
    )
    result = compute_character_divisor(presentation, character(*m))
    assert result.coefficients == tuple(
        sum(m_i * coordinate for m_i, coordinate in zip(m, ray, strict=True))
        for ray in presentation.rays
    )
    assert len(result.coefficients) == len(presentation.rays)


def test_ray_order_and_zero_coefficients_are_retained() -> None:
    result = compute_character_divisor(dp6_fan(), character(0, 5))
    # rays: (1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,-1)
    assert result.coefficients == (0, 5, 0, -5, 5, -5)
    assert result.support_ray_indices == (1, 3, 4, 5)


def test_invalid_fan_is_rejected_before_pairing() -> None:
    cones = tuple(cone for cone in DP6_CONES if cone != (3,))
    invalid = fan(
        ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1)), cones
    )
    with pytest.raises(OperationDomainValidationError) as rejection:
        compute_character_divisor(invalid, character(1, 1))
    assert rejection.value.errors()[0]["type"] == "toric.fan_not_valid"


def test_character_rank_must_match_the_lattice() -> None:
    with pytest.raises(ValidationError):
        CharacterDivisorRequest(
            fan=p2_fan(), character=character(1, 0, 0)
        )


def test_result_round_trips_through_strict_json() -> None:
    result = compute_character_divisor(p2_fan(), character(4, -7))
    replayed = CharacterDivisorResult.model_validate_json(result.model_dump_json())
    assert replayed == result
    assert replayed.coefficients == (4, -7, 3)


def test_catalog_and_native_paths_agree() -> None:
    payload = {
        "fan": {
            "lattice_rank": 2,
            "rays": [["1", "0"], ["0", "1"], ["-1", "-1"]],
            "cones": [[], [0], [1], [2], [0, 1], [0, 2], [1, 2]],
        },
        "character": {"entries": ["1", "1"]},
    }
    request = parse_operation_input(CharacterDivisorRequest, payload)
    catalog_result = CHARACTER_DIVISOR_OPERATION.run(request)
    native_result = compute_character_divisor(p2_fan(), character(1, 1))
    assert catalog_result == native_result
    assert catalog_result.coefficients == (1, 1, -2)
    assert len(P2_RAYS) == 3


def test_native_character_envelope_rejects_oversized_digits() -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError
    from jacobian.math.geometry.toric._models import CharacterVector

    oversized = CharacterVector.model_construct(entries=(10**9, 0))
    with pytest.raises(OperationResourceAdmissionError):
        compute_character_divisor(p2_fan(), oversized)
