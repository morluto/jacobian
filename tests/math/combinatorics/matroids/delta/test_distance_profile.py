"""Exact distance-to-feasibility profiles for finite delta-matroids."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids.delta import (
    DeltaMatroidDistanceProfile as PublicDeltaMatroidDistanceProfile,
)
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidDistanceProfileRequest,
)
from jacobian.math.combinatorics.matroids.delta._tools import _distance_profile
from jacobian.math.combinatorics.matroids.delta.values import FiniteDeltaMatroid


def _satisfies_symmetric_exchange(feasible_masks: tuple[int, ...]) -> bool:
    """Independent bit-set oracle for the complete symmetric-exchange axiom."""

    feasible = set(feasible_masks)
    for left in feasible_masks:
        for right in feasible_masks:
            difference = left ^ right
            remaining = difference
            while remaining:
                element_bit = remaining & -remaining
                remaining ^= element_bit
                candidates = difference
                exchange_exists = False
                while candidates:
                    candidate_bit = candidates & -candidates
                    candidates ^= candidate_bit
                    toggle = (
                        element_bit
                        if candidate_bit == element_bit
                        else element_bit | candidate_bit
                    )
                    if (left ^ toggle) in feasible:
                        exchange_exists = True
                        break
                if not exchange_exists:
                    return False
    return True


def _rows_from_mask(family_mask: int, ground_size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(
            tuple(index for index in range(ground_size) if subset_mask & (1 << index))
            for subset_mask in range(1 << ground_size)
            if family_mask & (1 << subset_mask)
        )
    )


def test_profile_matches_independent_oracle_for_every_small_delta_matroid() -> None:
    for ground_size in range(4):
        all_subset_masks = range(1 << ground_size)
        for family_mask in range(1, 1 << (1 << ground_size)):
            feasible_masks = tuple(
                mask for mask in all_subset_masks if family_mask & (1 << mask)
            )
            if not _satisfies_symmetric_exchange(feasible_masks):
                continue

            source = FiniteDeltaMatroid(
                ground=tuple(f"e{index}" for index in range(ground_size)),
                feasible=_rows_from_mask(family_mask, ground_size),
            )
            result = _distance_profile(
                DeltaMatroidDistanceProfileRequest(delta_matroid=source)
            )

            expected_distances: list[int] = []
            expected_nearest_counts: list[int] = []
            for subset_mask in all_subset_masks:
                differences = tuple(
                    (subset_mask ^ feasible_mask).bit_count()
                    for feasible_mask in feasible_masks
                )
                nearest_distance = min(differences)
                expected_distances.append(nearest_distance)
                expected_nearest_counts.append(differences.count(nearest_distance))
            expected_histogram = tuple(
                expected_distances.count(distance)
                for distance in range(ground_size + 1)
            )

            assert result.delta_matroid == source
            assert result.distance_by_mask == tuple(expected_distances)
            assert result.nearest_feasible_count_by_mask == tuple(
                expected_nearest_counts
            )
            assert result.distance_histogram == expected_histogram
            assert DeltaMatroidDistanceProfileRequest.model_validate_json(
                DeltaMatroidDistanceProfileRequest(
                    delta_matroid=source
                ).model_dump_json()
            ) == DeltaMatroidDistanceProfileRequest(delta_matroid=source)


def test_profile_rejects_inconsistent_histogram_and_coerced_integers() -> None:
    source = FiniteDeltaMatroid(ground=("a",), feasible=((),))
    payload = {
        "delta_matroid": source.model_dump(),
        "distance_by_mask": [0, 1],
        "nearest_feasible_count_by_mask": [1, 1],
        "distance_histogram": [0, 2],
    }
    with pytest.raises(ValueError, match="distance histogram must count"):
        PublicDeltaMatroidDistanceProfile.model_validate(payload)

    for invalid in (False, 0.0, "0"):
        malformed = {
            **payload,
            "distance_by_mask": [invalid, 1],
            "distance_histogram": [1, 1],
        }
        with pytest.raises(ValueError):
            PublicDeltaMatroidDistanceProfile.model_validate(malformed)


def test_profile_binds_nearest_counts_and_zero_masks_to_source() -> None:
    source = FiniteDeltaMatroid(ground=("a",), feasible=((),))
    for distances, counts in (((1, 0), (1, 1)), ((0, 1), (2, 1))):
        with pytest.raises(ValueError):
            PublicDeltaMatroidDistanceProfile.model_validate(
                {
                    "delta_matroid": source.model_dump(),
                    "distance_by_mask": distances,
                    "nearest_feasible_count_by_mask": counts,
                    "distance_histogram": (1, 1),
                }
            )


def test_empty_ground_profile_and_json_round_trip() -> None:
    source = FiniteDeltaMatroid(ground=(), feasible=((),))
    result = _distance_profile(DeltaMatroidDistanceProfileRequest(delta_matroid=source))

    assert result.distance_by_mask == (0,)
    assert type(result) is PublicDeltaMatroidDistanceProfile
    assert result.nearest_feasible_count_by_mask == (1,)
    assert result.distance_histogram == (1,)
    assert type(result).model_validate_json(result.model_dump_json()) == result




def test_native_malformed_profile_source_is_domain_error() -> None:
    from jacobian.math.combinatorics.matroids.delta.operations import distance_profile

    malformed = FiniteDeltaMatroid.model_construct()
    with pytest.raises(OperationDomainValidationError) as error:
        distance_profile(malformed)
    assert error.value.errors()[0]["type"] == "delta_matroid.source_not_valid"


def test_forged_non_delta_source_is_rejected() -> None:
    source = FiniteDeltaMatroid(ground=("a", "b", "c"), feasible=((), (0, 1), (2,)))

    with pytest.raises(OperationDomainValidationError) as error:
        _distance_profile(DeltaMatroidDistanceProfileRequest(delta_matroid=source))

    assert error.value.errors()[0]["type"] == "delta_matroid.source_not_valid"


def test_complete_subset_state_boundary_is_accepted() -> None:
    source = FiniteDeltaMatroid(
        ground=tuple(f"e{index}" for index in range(12)), feasible=((),)
    )
    result = _distance_profile(DeltaMatroidDistanceProfileRequest(delta_matroid=source))

    assert len(result.distance_by_mask) == 4_096
    assert result.distance_by_mask[-1] == 12
    assert result.distance_histogram == tuple(
        math.comb(12, distance) for distance in range(13)
    )


def test_ground_subset_state_limit_is_preflighted() -> None:
    source = FiniteDeltaMatroid(
        ground=tuple(f"e{index}" for index in range(13)), feasible=((),)
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        _distance_profile(DeltaMatroidDistanceProfileRequest(delta_matroid=source))

    assert error.value.errors()[0]["type"] == (
        "delta_matroid.distance_profile_states_exceeded"
    )


def test_raw_feasible_membership_limit_is_preflighted_before_nested_parsing() -> None:
    raw = {"delta_matroid": {"ground": ["a"], "feasible": [[0]] * 20_000}}

    with pytest.raises(ValidationError) as error:
        DeltaMatroidDistanceProfileRequest.model_validate(raw)

    assert error.value.errors()[0]["type"] == (
        "delta_matroid.source_memberships_exceeded"
    )


def test_subset_by_feasible_family_evaluations_are_preflighted() -> None:
    # All subsets of the first seven elements form a delta-matroid. Its
    # exchange admission is below 250,000, while its profile would require
    # 4,096 * 128 distance evaluations.
    source = FiniteDeltaMatroid(
        ground=tuple(f"e{index}" for index in range(12)),
        feasible=tuple(
            sorted(
                tuple(index for index in range(7) if subset_mask & (1 << index))
                for subset_mask in range(1 << 7)
            )
        ),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        _distance_profile(DeltaMatroidDistanceProfileRequest(delta_matroid=source))

    assert error.value.errors()[0]["type"] == (
        "delta_matroid.distance_profile_work_exceeded"
    )
