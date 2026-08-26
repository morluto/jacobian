"""Tests for complete indexed subset-sum multiplicity profiles."""

from __future__ import annotations

from collections import Counter
from itertools import product

import pytest
from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from jacobian.canonical import (
    CanonicalLimits,
    canonicalize_json,
    format_canonical_integer,
)
from jacobian.math.additive_combinatorics import (
    IndexedIntegerSequence,
    SubsetSumProfile,
    SubsetSumProfileEntry,
    subset_sum_profile,
)
from jacobian.math.additive_combinatorics._models import SubsetSumProfileRequest
from jacobian.math.additive_combinatorics._operations import (
    compute_subset_sum_profile,
    verify_subset_sum_profile,
)
from jacobian.math.additive_combinatorics.operations import (
    MAX_SUBSET_SUM_DP_TRANSITIONS,
    MAX_SUBSET_SUM_PROFILE_RESULT_BYTES,
    _subset_sum_profile_envelope,
)
from jacobian.math.additive_combinatorics.values import (
    MAX_SUBSET_SUM_ITEM_DIGITS,
    MAX_SUBSET_SUM_ITEMS,
    MAX_SUBSET_SUM_SUM_DIGITS,
)


def _request(*items: int) -> SubsetSumProfileRequest:
    return SubsetSumProfileRequest(
        source={"items": [format_canonical_integer(item) for item in items]},
    )


def _numeric_profile(result: SubsetSumProfile) -> dict[int, int]:
    return {int(entry.sum): int(entry.multiplicity) for entry in result.entries}


def _bitmask_oracle(items: tuple[int, ...]) -> Counter[int]:
    return Counter(
        sum(item for index, item in enumerate(items) if mask & (1 << index))
        for mask in range(1 << len(items))
    )


def test_repeated_values_remain_distinct_indexed_items() -> None:
    result = compute_subset_sum_profile(_request(1, 1))

    assert _numeric_profile(result) == {0: 1, 1: 2, 2: 1}
    assert result.total_subsets == "4"
    assert result.support_size == 3


def test_empty_source_contains_exactly_the_empty_subset() -> None:
    result = compute_subset_sum_profile(_request())

    assert _numeric_profile(result) == {0: 1}
    assert result.source.items == ()
    assert result.total_subsets == "1"


def test_zeros_change_multiplicity_without_changing_support() -> None:
    result = compute_subset_sum_profile(_request(0, 0, 2))

    assert _numeric_profile(result) == {0: 4, 2: 4}
    assert result.total_subsets == "8"


def test_mixed_sign_profile_is_sorted_and_complete() -> None:
    result = compute_subset_sum_profile(_request(-1, 2))

    assert tuple(int(entry.sum) for entry in result.entries) == (-1, 0, 1, 2)
    assert all(entry.multiplicity == "1" for entry in result.entries)


def test_small_profiles_match_independent_bitmask_enumeration() -> None:
    for length in range(5):
        for items in product(range(-2, 3), repeat=length):
            result = compute_subset_sum_profile(_request(*items))
            assert _numeric_profile(result) == _bitmask_oracle(items)


def test_permuting_source_preserves_numeric_profile() -> None:
    forward = compute_subset_sum_profile(_request(-3, 1, 1, 4))
    permuted = compute_subset_sum_profile(_request(1, 4, -3, 1))

    assert _numeric_profile(forward) == _numeric_profile(permuted)
    assert forward.source != permuted.source


def test_conway_guy_eleven_set_has_2048_distinct_subset_sums() -> None:
    # Source-backed convention fixture: the Conway-Guy upper witness recorded
    # in erdos-frontier-atlas P1 at revision 0394e3d3b249439ffabec7d96a3311aa441651b8.
    source = (285, 433, 510, 550, 570, 581, 587, 590, 592, 593, 594)

    result = compute_subset_sum_profile(_request(*source))

    assert result.support_size == 2048
    assert result.total_subsets == "2048"
    assert all(entry.multiplicity == "1" for entry in result.entries)


def test_result_round_trip_has_complete_profile_verifier() -> None:
    result = compute_subset_sum_profile(_request(-2, 0, 3, 3))

    decoded = SubsetSumProfile.model_validate(result.model_dump())
    assert decoded == result
    assert verify_subset_sum_profile(decoded)


def test_verifier_rejects_mutated_source() -> None:
    result = compute_subset_sum_profile(_request(1, 1))
    payload = result.model_dump(mode="json")
    payload["source"]["items"] = ["1", "2"]

    assert not verify_subset_sum_profile(SubsetSumProfile.model_validate(payload))


def test_verifier_rejects_mutated_multiplicity() -> None:
    result = compute_subset_sum_profile(_request(1, 1))
    payload = result.model_dump(mode="json")
    payload["entries"][1]["multiplicity"] = "3"

    assert not verify_subset_sum_profile(SubsetSumProfile.model_validate(payload))


def test_verifier_rejects_mutated_profile_sum() -> None:
    result = compute_subset_sum_profile(_request(1, 1))
    payload = result.model_dump(mode="json")
    payload["entries"][-1]["sum"] = "3"

    assert not verify_subset_sum_profile(SubsetSumProfile.model_validate(payload))


def test_verifier_rejects_mutated_total() -> None:
    result = compute_subset_sum_profile(_request(1, 1))
    payload = result.model_dump(mode="json")
    payload["total_subsets"] = "3"

    assert not verify_subset_sum_profile(SubsetSumProfile.model_validate(payload))


def test_result_sensitive_admission_accepts_many_repeated_zeros() -> None:
    source = IndexedIntegerSequence(items=("0",) * MAX_SUBSET_SUM_ITEMS)

    result = subset_sum_profile(source)

    assert _numeric_profile(result) == {0: 1 << MAX_SUBSET_SUM_ITEMS}
    assert result.total_subsets == str(1 << MAX_SUBSET_SUM_ITEMS)


def test_widened_source_contract_admits_beyond_legacy_multiplicity_digits() -> None:
    source = IndexedIntegerSequence(items=("0",) * 300)

    result = subset_sum_profile(source)

    assert _numeric_profile(result) == {0: 1 << 300}
    assert result.total_subsets == str(1 << 300)
    assert len(result.total_subsets) > len(str(1 << 256))


def test_profile_work_above_bound_is_rejected_before_execution() -> None:
    items = tuple(1 << exponent for exponent in range(14)) + (0,) * (
        MAX_SUBSET_SUM_ITEMS - 14
    )

    with pytest.raises(ValidationError):
        _request(*items)


def test_profile_result_above_bound_is_rejected_before_execution() -> None:
    offset = 10 ** (MAX_SUBSET_SUM_ITEM_DIGITS - 1)
    items = tuple(offset + (1 << exponent) for exponent in range(15))

    with pytest.raises(ValidationError):
        _request(*items)


def test_large_accepted_profile_stays_inside_declared_result_budget() -> None:
    offset = 10 ** (MAX_SUBSET_SUM_ITEM_DIGITS - 1)
    source = tuple(offset + (1 << exponent) for exponent in range(6))
    result = compute_subset_sum_profile(_request(*source))

    encoded = canonicalize_json(
        result.model_dump(mode="json"),
        limits=CanonicalLimits(
            max_output_bytes=MAX_SUBSET_SUM_PROFILE_RESULT_BYTES,
        ),
    )

    assert result.support_size == 1 << 6
    assert len(encoded) <= MAX_SUBSET_SUM_PROFILE_RESULT_BYTES


def test_source_digit_bound_is_enforced_before_integer_conversion() -> None:
    with pytest.raises(ValidationError):
        SubsetSumProfileRequest(
            source={"items": ["9" * (MAX_SUBSET_SUM_ITEM_DIGITS + 1)]},
        )

    with pytest.raises(ValidationError) as error:
        SubsetSumProfileRequest(source={"items": ["9" * 100_000]})
    assert error.value.errors()[0]["type"] == "string_too_long"


def test_source_item_count_bound_is_enforced_by_admission() -> None:
    widened = IndexedIntegerSequence(items=("0",) * (MAX_SUBSET_SUM_ITEMS + 1))
    assert len(widened.items) == MAX_SUBSET_SUM_ITEMS + 1

    with pytest.raises(ValidationError):
        SubsetSumProfileRequest(source=widened)

    with pytest.raises(ValidationError):
        IndexedIntegerSequence(items=("0",) * (500_000 + 1))


def test_profile_envelope_rejects_oversized_sources_before_integer_conversion() -> None:
    widened = IndexedIntegerSequence(items=("0",) * (MAX_SUBSET_SUM_ITEMS + 1))

    with pytest.raises(ValueError):
        _subset_sum_profile_envelope(widened)


def test_raw_item_count_bound_is_enforced_before_nested_parsing() -> None:
    payload = {"source": {"items": ["z"] * (MAX_SUBSET_SUM_ITEMS + 1)}}

    with pytest.raises(ValidationError):
        SubsetSumProfileRequest.model_validate(payload)


def test_wide_canonical_items_are_rejected_by_the_raw_item_count_bound() -> None:
    payload = {
        "source": {"items": ["9" * 1_000] * (MAX_SUBSET_SUM_ITEMS + 1)},
    }

    with pytest.raises(ValidationError):
        SubsetSumProfileRequest.model_validate(payload)


def test_oversized_json_list_is_rejected_before_nested_item_parsing() -> None:
    payload = {"source": {"items": ["z", *(["1"] * MAX_SUBSET_SUM_ITEMS)]}}

    with pytest.raises(ValidationError) as error:
        SubsetSumProfileRequest.model_validate(payload)

    assert (
        f"{MAX_SUBSET_SUM_ITEMS:,}-item profile bound" in error.value.errors()[0]["msg"]
    )


def test_oversized_tuple_container_is_rejected_before_nested_item_parsing() -> None:
    payload = {"source": {"items": ("z", *(["1"] * MAX_SUBSET_SUM_ITEMS))}}

    with pytest.raises(ValidationError) as error:
        SubsetSumProfileRequest.model_validate(payload)

    assert (
        f"{MAX_SUBSET_SUM_ITEMS:,}-item profile bound" in error.value.errors()[0]["msg"]
    )


def test_expensive_admissible_items_are_rejected_by_the_raw_preflight_bound() -> None:
    payload = {"source": {"items": ("9" * 2_048,) * (MAX_SUBSET_SUM_ITEMS + 1)}}

    with pytest.raises(PydanticCustomError) as error:
        SubsetSumProfileRequest.bound_raw_source(payload)

    assert error.value.type == "additive_combinatorics.bound_raw_source"


def test_json_list_at_the_item_ceiling_remains_admitted_and_canonical() -> None:
    admitted = SubsetSumProfileRequest.model_validate(
        {"source": {"items": ["0"] * MAX_SUBSET_SUM_ITEMS}}
    )

    assert admitted.source.items == ("0",) * MAX_SUBSET_SUM_ITEMS


@pytest.mark.parametrize(
    ("prefix", "message"),
    [
        ("", "source-sum digit bound"),
        ("-", f"at most {MAX_SUBSET_SUM_SUM_DIGITS + 1} characters"),
    ],
)
def test_profile_entry_sum_digit_bound_applies_to_either_sign(
    prefix: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError):
        SubsetSumProfileEntry(
            sum=prefix + "9" * (MAX_SUBSET_SUM_SUM_DIGITS + 1),
            multiplicity="1",
        )


def test_request_schema_exposes_source_shape_and_character_bounds() -> None:
    schema = SubsetSumProfileRequest.model_json_schema()
    items_schema = schema["properties"]["source"]["properties"]["items"]
    source_description = schema["properties"]["source"]["description"]

    assert items_schema["maxItems"] == MAX_SUBSET_SUM_ITEMS
    assert items_schema["items"]["maxLength"] == MAX_SUBSET_SUM_ITEM_DIGITS + 1
    assert "4*n*S" in source_description
    assert f"{MAX_SUBSET_SUM_DP_TRANSITIONS:,}" in source_description
    assert f"{MAX_SUBSET_SUM_PROFILE_RESULT_BYTES:,} bytes" in source_description


def test_schema_item_ceiling_matches_validator_at_the_boundary() -> None:
    at_ceiling = IndexedIntegerSequence(items=("0",) * MAX_SUBSET_SUM_ITEMS)
    admitted = SubsetSumProfileRequest(source=at_ceiling)
    assert len(admitted.source.items) == MAX_SUBSET_SUM_ITEMS

    beyond = IndexedIntegerSequence(items=("0",) * (MAX_SUBSET_SUM_ITEMS + 1))
    with pytest.raises(ValidationError):
        SubsetSumProfileRequest(source=beyond)
