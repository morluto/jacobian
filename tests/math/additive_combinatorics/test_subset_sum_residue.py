"""Tests for complete modular subset-sum profiles."""

from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.canonical import CanonicalLimits, canonicalize_json
from jacobian.math.additive_combinatorics import (
    IndexedIntegerSequence,
    IndexSubset,
)
from jacobian.math.additive_combinatorics._subset_sum_residue import (
    MAX_RESIDUE_PROFILE_DP_CELLS,
    MAX_RESIDUE_PROFILE_INPUT_INTEGER_DIGITS,
    MAX_RESIDUE_PROFILE_MODULUS,
    MAX_RESIDUE_PROFILE_MULTIPLICITY_BITS,
    MAX_RESIDUE_PROFILE_RESULT_BYTES,
    SubsetSumResidueProfileRequest,
    SubsetSumResidueProfileResult,
    compute_subset_sum_residue_profile,
)


def _request(
    values: tuple[int, ...],
    modulus: int,
    *,
    include_empty_subset: bool,
    include_witnesses: bool = False,
) -> SubsetSumResidueProfileRequest:
    return SubsetSumResidueProfileRequest(
        source={"items": [str(value) for value in values]},
        modulus=modulus,
        include_empty_subset=include_empty_subset,
        include_witnesses=include_witnesses,
    )


def _bitmask_oracle(
    values: tuple[int, ...],
    modulus: int,
    *,
    include_empty_subset: bool,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...] | None, ...]]:
    counts = [0] * modulus
    masks: list[int | None] = [None] * modulus
    for mask in range(1 << len(values)):
        if mask == 0 and not include_empty_subset:
            continue
        residue = (
            sum(value for index, value in enumerate(values) if mask & (1 << index))
            % modulus
        )
        counts[residue] += 1
        if masks[residue] is None or mask < masks[residue]:
            masks[residue] = mask
    witnesses = tuple(
        None
        if mask is None
        else tuple(index for index in range(len(values)) if mask & (1 << index))
        for mask in masks
    )
    return tuple(counts), witnesses


def test_request_and_result_compose_through_strict_json_parsing() -> None:
    request = SubsetSumResidueProfileRequest.model_validate_json(
        '{"source":{"items":["2","3"]},"modulus":5,'
        '"include_empty_subset":false,"include_witnesses":true}',
        strict=True,
    )

    assert request.source == IndexedIntegerSequence(items=("2", "3"))
    result = compute_subset_sum_residue_profile(request)
    assert (
        SubsetSumResidueProfileResult.model_validate_json(
            result.model_dump_json(), strict=True
        )
        == result
    )


def test_two_items_have_nonempty_zero_residue_with_canonical_witnesses() -> None:
    result = compute_subset_sum_residue_profile(
        _request(
            (2, 3),
            5,
            include_empty_subset=False,
            include_witnesses=True,
        )
    )

    assert result.residue_counts == ("1", "0", "1", "1", "0")
    assert result.residue_witnesses == (
        IndexSubset(indices=(0, 1)),
        None,
        IndexSubset(indices=(0,)),
        IndexSubset(indices=(1,)),
        None,
    )


@pytest.mark.parametrize(
    ("include_empty_subset", "expected"),
    [(True, ("1", "0", "0", "0")), (False, ("0", "0", "0", "0"))],
)
def test_empty_source_has_explicit_empty_subset_convention(
    include_empty_subset: bool,
    expected: tuple[str, ...],
) -> None:
    result = compute_subset_sum_residue_profile(
        _request((), 4, include_empty_subset=include_empty_subset)
    )
    assert result.residue_counts == expected
    assert result.source == IndexedIntegerSequence(items=())


def test_single_item_does_not_claim_nonempty_zero_residue() -> None:
    result = compute_subset_sum_residue_profile(
        _request((2,), 5, include_empty_subset=False, include_witnesses=True)
    )
    assert result.residue_counts[0] == "0"
    assert result.residue_witnesses is not None
    assert result.residue_witnesses[0] is None


def test_problem_131_local_predicate_and_planted_failure() -> None:
    def zero_counts_after_removal(values: tuple[int, ...]) -> tuple[int, ...]:
        counts: list[int] = []
        for removed_index, modulus in enumerate(values):
            remaining = values[:removed_index] + values[removed_index + 1 :]
            result = compute_subset_sum_residue_profile(
                _request(
                    remaining,
                    modulus,
                    include_empty_subset=False,
                )
            )
            counts.append(int(result.residue_counts[0]))
        return tuple(counts)

    assert zero_counts_after_removal((2, 3)) == (0, 0)
    assert zero_counts_after_removal((2, 3, 5)) == (1, 0, 1)


def test_repeated_zeros_preserve_index_multiplicity() -> None:
    result = compute_subset_sum_residue_profile(
        _request((0, 0), 3, include_empty_subset=False, include_witnesses=True)
    )
    assert result.residue_counts == ("3", "0", "0")
    assert result.residue_witnesses is not None
    assert result.residue_witnesses[0] == IndexSubset(indices=(0,))


def test_negative_values_are_reduced_only_for_the_recurrence() -> None:
    request = _request((-7, 3), 5, include_empty_subset=True)
    result = compute_subset_sum_residue_profile(request)
    assert result.source == request.source
    assert result.residue_counts == ("1", "1", "0", "2", "0")


def test_translation_by_modulus_multiples_preserves_profile() -> None:
    base = compute_subset_sum_residue_profile(
        _request((-2, 0, 3), 7, include_empty_subset=False)
    )
    translated = compute_subset_sum_residue_profile(
        _request((12, -21, 24), 7, include_empty_subset=False)
    )
    assert translated.residue_counts == base.residue_counts


def test_all_small_profiles_match_complete_bitmask_enumeration() -> None:
    for item_count in range(5):
        for values in product(range(-2, 3), repeat=item_count):
            for modulus in range(1, 6):
                for include_empty_subset in (False, True):
                    result = compute_subset_sum_residue_profile(
                        _request(
                            values,
                            modulus,
                            include_empty_subset=include_empty_subset,
                            include_witnesses=True,
                        )
                    )
                    expected_counts, expected_witnesses = _bitmask_oracle(
                        values,
                        modulus,
                        include_empty_subset=include_empty_subset,
                    )
                    assert tuple(map(int, result.residue_counts)) == expected_counts
                    assert result.residue_witnesses is not None
                    assert (
                        tuple(
                            None if witness is None else witness.indices
                            for witness in result.residue_witnesses
                        )
                        == expected_witnesses
                    )


@pytest.mark.parametrize("include_empty_subset", [False, True])
def test_total_multiplicity_is_the_number_of_permitted_subsets(
    include_empty_subset: bool,
) -> None:
    values = (0, 1, 1, 5, -3, 12)
    result = compute_subset_sum_residue_profile(
        _request(values, 8, include_empty_subset=include_empty_subset)
    )
    expected_total = (1 << len(values)) - (not include_empty_subset)
    assert sum(map(int, result.residue_counts)) == expected_total


def test_result_rejects_mutated_count() -> None:
    result = compute_subset_sum_residue_profile(
        _request((1, 2, 4), 5, include_empty_subset=False)
    )
    payload = result.model_dump(mode="json")
    payload["residue_counts"][0] = "8"
    with pytest.raises(ValidationError, match="residue counts do not match"):
        SubsetSumResidueProfileResult.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source", {"items": ["1", "2", "3"]}),
        ("modulus", 4),
        ("include_empty_subset", True),
        ("include_witnesses", False),
    ],
)
def test_result_rejects_mutated_source_relation(
    field: str,
    replacement: object,
) -> None:
    result = compute_subset_sum_residue_profile(
        _request(
            (1, 2, 4),
            5,
            include_empty_subset=False,
            include_witnesses=True,
        )
    )
    payload = result.model_dump(mode="json")
    payload[field] = replacement
    with pytest.raises(ValidationError):
        SubsetSumResidueProfileResult.model_validate(payload)


def test_result_rejects_source_reordering_that_changes_canonical_witnesses() -> None:
    result = compute_subset_sum_residue_profile(
        _request(
            (1, 2, 4),
            5,
            include_empty_subset=False,
            include_witnesses=True,
        )
    )
    payload = result.model_dump(mode="json")
    payload["source"] = {"items": ["4", "2", "1"]}
    with pytest.raises(ValidationError, match="residue witnesses do not match"):
        SubsetSumResidueProfileResult.model_validate(payload)


def test_result_rejects_noncanonical_or_mutated_witness() -> None:
    result = compute_subset_sum_residue_profile(
        _request(
            (0, 0),
            3,
            include_empty_subset=False,
            include_witnesses=True,
        )
    )
    payload = result.model_dump(mode="json")
    payload["residue_witnesses"][0] = {"indices": [1]}
    with pytest.raises(ValidationError, match="residue witnesses do not match"):
        SubsetSumResidueProfileResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["residue_witnesses"][0] = {"indices": [1, 0]}
    with pytest.raises(ValidationError, match="strictly increasing"):
        SubsetSumResidueProfileResult.model_validate(payload)


def test_request_rejects_one_oversized_source_integer_before_parsing() -> None:
    oversized = "1" + "0" * MAX_RESIDUE_PROFILE_INPUT_INTEGER_DIGITS
    with pytest.raises(ValidationError, match="32,768-digit input bound"):
        SubsetSumResidueProfileRequest(
            source={"items": [oversized]},
            modulus=2,
            include_empty_subset=True,
        )


def test_request_rejects_multiplicity_intermediate_above_bound() -> None:
    with pytest.raises(ValidationError, match="4,096-bit intermediate bound"):
        SubsetSumResidueProfileRequest(
            source={"items": ["0"] * MAX_RESIDUE_PROFILE_MULTIPLICITY_BITS},
            modulus=1,
            include_empty_subset=True,
        )


def test_request_bounds_raw_source_before_nested_parsing() -> None:
    with pytest.raises(ValidationError, match="4,096-bit intermediate bound"):
        SubsetSumResidueProfileRequest.model_validate(
            {
                "source": {
                    "items": ["not-an-integer"] * MAX_RESIDUE_PROFILE_MULTIPLICITY_BITS
                },
                "modulus": 1,
                "include_empty_subset": True,
            }
        )


def test_source_schema_admits_the_advertised_residue_envelope() -> None:
    schema = SubsetSumResidueProfileRequest.model_json_schema()
    items_schema = schema["$defs"]["IndexedIntegerSequence"]["properties"]["items"]

    assert items_schema["maxItems"] == MAX_RESIDUE_PROFILE_MULTIPLICITY_BITS - 1
    assert (
        items_schema["items"]["maxLength"]
        == MAX_RESIDUE_PROFILE_INPUT_INTEGER_DIGITS + 1
    )


def test_positions_beyond_legacy_source_cap_are_admitted() -> None:
    result = compute_subset_sum_residue_profile(
        _request((0,) * 257, 1, include_empty_subset=False)
    )

    assert result.residue_counts == (str((1 << 257) - 1),)


def test_single_integer_beyond_legacy_digit_cap_is_admitted() -> None:
    value = 10**256

    result = compute_subset_sum_residue_profile(
        _request((value,), 7, include_empty_subset=False)
    )

    assert result.residue_counts == tuple(
        "1" if residue == value % 7 else "0" for residue in range(7)
    )


def test_exact_dp_cell_boundary_is_complete_and_serializable() -> None:
    item_count = 200
    modulus = MAX_RESIDUE_PROFILE_DP_CELLS // item_count
    request = SubsetSumResidueProfileRequest(
        source={"items": ["0"] * item_count},
        modulus=modulus,
        include_empty_subset=True,
    )
    result = compute_subset_sum_residue_profile(request)
    assert result.residue_counts[0] == str(1 << item_count)
    assert set(result.residue_counts[1:]) == {"0"}
    encoded = canonicalize_json(
        result.model_dump(mode="json"),
        limits=CanonicalLimits(max_output_bytes=MAX_RESIDUE_PROFILE_RESULT_BYTES),
    )
    assert len(encoded) <= MAX_RESIDUE_PROFILE_RESULT_BYTES


def test_request_just_above_dp_cell_boundary_is_rejected() -> None:
    item_count = 201
    modulus = MAX_RESIDUE_PROFILE_DP_CELLS // 200
    with pytest.raises(ValidationError, match="1,000,000-cell work bound"):
        SubsetSumResidueProfileRequest(
            source={"items": ["0"] * item_count},
            modulus=modulus,
            include_empty_subset=True,
        )


def test_modulus_boundary_and_schema_are_explicit() -> None:
    request = _request(
        (),
        MAX_RESIDUE_PROFILE_MODULUS,
        include_empty_subset=False,
    )
    assert request.modulus == MAX_RESIDUE_PROFILE_MODULUS

    with pytest.raises(ValidationError):
        _request(
            (),
            MAX_RESIDUE_PROFILE_MODULUS + 1,
            include_empty_subset=False,
        )

    schema = SubsetSumResidueProfileRequest.model_json_schema()
    assert schema["properties"]["modulus"]["maximum"] == (MAX_RESIDUE_PROFILE_MODULUS)
    assert "include_empty_subset" in schema["required"]
    source_schema = schema["$defs"]["IndexedIntegerSequence"]
    assert (
        "distinct indexed items" in source_schema["properties"]["items"]["description"]
    )

    with pytest.raises(ValidationError):
        SubsetSumResidueProfileRequest(
            source={"items": []},
            modulus=2,
            include_empty_subset=1,
        )


def test_witness_and_result_output_budgets_reject_before_work() -> None:
    with pytest.raises(ValidationError, match="index-slot storage bound"):
        SubsetSumResidueProfileRequest(
            source={"items": ["0"] * 251},
            modulus=1000,
            include_empty_subset=False,
            include_witnesses=True,
        )


def test_result_bounds_raw_arrays_before_source_binding_replay() -> None:
    base = compute_subset_sum_residue_profile(
        _request((), 4, include_empty_subset=True, include_witnesses=True)
    ).model_dump(mode="json")

    too_many_counts = dict(base)
    too_many_counts["residue_counts"] = ["0"] * 5
    with pytest.raises(ValidationError, match="exactly modulus rows"):
        SubsetSumResidueProfileResult.model_validate(too_many_counts)

    oversized_count = dict(base)
    oversized_count["residue_counts"] = ["10", "0", "0", "0"]
    with pytest.raises(ValidationError, match="source-derived multiplicity bound"):
        SubsetSumResidueProfileResult.model_validate(oversized_count)

    too_many_witnesses = dict(base)
    too_many_witnesses["residue_witnesses"] = [None] * 5
    with pytest.raises(ValidationError, match="exactly modulus rows"):
        SubsetSumResidueProfileResult.model_validate(too_many_witnesses)

    out_of_range_witness = dict(base)
    out_of_range_witness["residue_witnesses"] = [
        {"indices": [0]},
        None,
        None,
        None,
    ]
    with pytest.raises(ValidationError, match="retained source length"):
        SubsetSumResidueProfileResult.model_validate(out_of_range_witness)

    one_item = compute_subset_sum_residue_profile(
        _request((0,), 4, include_empty_subset=True, include_witnesses=True)
    ).model_dump(mode="json")
    one_item["residue_witnesses"][0] = {"indices": [1]}
    with pytest.raises(ValidationError, match="outside the retained source"):
        SubsetSumResidueProfileResult.model_validate(one_item)

    widest = "1" + "0" * (MAX_RESIDUE_PROFILE_INPUT_INTEGER_DIGITS - 1)
    with pytest.raises(ValidationError, match="4 MiB result bound"):
        SubsetSumResidueProfileRequest(
            source={"items": [widest] * 128},
            modulus=1,
            include_empty_subset=False,
        )


def test_shared_values_reject_ambiguous_index_subsets() -> None:
    assert IndexedIntegerSequence(items=("1", "1", "0")).items == (
        "1",
        "1",
        "0",
    )
    with pytest.raises(ValidationError, match="strictly increasing"):
        IndexSubset(indices=(0, 0))
    with pytest.raises(ValidationError):
        IndexSubset(indices=(-1,))
    with pytest.raises(ValidationError):
        IndexSubset.model_validate({"indices": ["0"]})
