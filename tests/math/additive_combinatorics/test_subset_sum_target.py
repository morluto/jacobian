from __future__ import annotations

import re
from itertools import permutations, product
from typing import cast

import pytest
from pydantic import TypeAdapter, ValidationError

from jacobian.catalog.models import MathTool
from jacobian.math.additive_combinatorics import (
    IndexedIntegerSequence,
    IndexSubset,
    subset_sum_profile,
)
from jacobian.math.additive_combinatorics._subset_sum_residue import (
    SubsetSumResidueProfileRequest,
    compute_subset_sum_residue_profile,
)
from jacobian.math.additive_combinatorics._subset_sum_target import (
    MAX_SUBSET_SUM_COMPLETE_CALL_PASSES,
    MAX_SUBSET_SUM_INTEGER_DIGITS,
    MAX_SUBSET_SUM_RECONSTRUCTED_DIGITS,
    MAX_SUBSET_SUM_TOTAL_TRANSITIONS,
    MAX_SUBSET_SUM_TRANSITIONS_PER_PASS,
    SubsetSumTargetRequest,
    SubsetSumTargetResult,
    _SubsetSumTargetScalar,
)
from jacobian.math.additive_combinatorics._subset_sum_target_kernel import (
    _solve_subset_sum_target,
)
from jacobian.math.additive_combinatorics._tools import (
    ADDITIVE_COMBINATORICS_OPERATIONS,
)
from jacobian.math.additive_combinatorics.values import MAX_SUBSET_SUM_ITEMS


def _operation() -> MathTool[SubsetSumTargetRequest, SubsetSumTargetResult]:
    return cast(
        MathTool[SubsetSumTargetRequest, SubsetSumTargetResult],
        next(
            operation
            for operation in ADDITIVE_COMBINATORICS_OPERATIONS
            if operation.operation_id == "additive.subset_sum.target.solve"
        ),
    )


def _request(
    values: tuple[int, ...],
    target: int,
    *,
    allow_empty_subset: bool,
) -> SubsetSumTargetRequest:
    return SubsetSumTargetRequest(
        source=IndexedIntegerSequence(items=tuple(str(value) for value in values)),
        target=str(target),
        allow_empty_subset=allow_empty_subset,
    )


def _brute_force(
    values: tuple[int, ...],
    target: int,
    *,
    allow_empty_subset: bool,
) -> tuple[int, ...] | None:
    for mask in range(1 << len(values)):
        if mask == 0 and not allow_empty_subset:
            continue
        indices = tuple(index for index in range(len(values)) if mask & (1 << index))
        if sum(values[index] for index in indices) == target:
            return indices
    return None


def test_request_and_result_compose_through_strict_json_parsing() -> None:
    request = SubsetSumTargetRequest.model_validate_json(
        '{"source":{"items":["2","3"]},"target":"5","allow_empty_subset":false}',
        strict=True,
    )

    assert request.source == IndexedIntegerSequence(items=("2", "3"))
    result = _operation().run(request)
    assert (
        SubsetSumTargetResult.model_validate_json(result.model_dump_json(), strict=True)
        == result
    )


@pytest.mark.parametrize(
    ("items", "target", "allow_empty_subset", "expected"),
    (
        ((2, 3), 5, False, (0, 1)),
        ((2, 3), 4, False, None),
        ((1, 1), 2, False, (0, 1)),
        ((), 0, True, ()),
        ((), 0, False, None),
        ((0,), 0, True, ()),
        ((0,), 0, False, (0,)),
        ((4, -7, 3), -4, False, (1, 2)),
        ((2, 3), 0, False, None),
        ((-2, -3), 0, False, None),
        ((0, 3), 0, False, (0,)),
    ),
)
def test_target_kernel_known_answers(
    items: tuple[int, ...],
    target: int,
    allow_empty_subset: bool,
    expected: tuple[int, ...] | None,
) -> None:
    assert (
        _solve_subset_sum_target(
            items,
            target,
            allow_empty_subset=allow_empty_subset,
        )
        == expected
    )


def test_operation_reports_attained_and_not_attained() -> None:
    attained = _operation().run(_request((2, 3), 5, allow_empty_subset=False))
    assert attained == SubsetSumTargetResult(
        source=IndexedIntegerSequence(items=("2", "3")),
        target="5",
        allow_empty_subset=False,
        status="ATTAINED",
        witness=IndexSubset(indices=(0, 1)),
        reconstructed_sum="5",
    )

    not_attained = _operation().run(_request((2, 3), 4, allow_empty_subset=False))
    assert not_attained == SubsetSumTargetResult(
        source=IndexedIntegerSequence(items=("2", "3")),
        target="4",
        allow_empty_subset=False,
        status="NOT_ATTAINED",
    )


def test_canonical_witness_uses_the_smallest_incidence_mask() -> None:
    # Both {0,1} and {2} sum to five; masks 0b011 and 0b100 make {0,1}
    # the stable canonical witness.
    result = _operation().run(_request((3, 2, 5), 5, allow_empty_subset=False))
    assert result.witness == IndexSubset(indices=(0, 1))


def test_proper_divisors_of_seventy_do_not_attain_seventy() -> None:
    result = _operation().run(
        _request((1, 2, 5, 7, 10, 14, 35), 70, allow_empty_subset=False)
    )
    assert result.status == "NOT_ATTAINED"


def test_solver_agrees_with_exhaustive_bitmasks() -> None:
    for size in range(5):
        for values in product(range(-2, 3), repeat=size):
            for target in range(-5, 6):
                for allow_empty_subset in (False, True):
                    assert _solve_subset_sum_target(
                        values,
                        target,
                        allow_empty_subset=allow_empty_subset,
                    ) == _brute_force(
                        values,
                        target,
                        allow_empty_subset=allow_empty_subset,
                    )


def test_input_permutations_preserve_status_and_transport_a_witness() -> None:
    for values in permutations((4, -1, 2)):
        result = _operation().run(_request(values, 3, allow_empty_subset=False))
        assert result.status == "ATTAINED"
        assert result.witness is not None
        assert sum(values[index] for index in result.witness.indices) == 3


def test_result_rejects_source_decision_and_witness_mutations() -> None:
    valid = _operation().run(_request((3, 2, 5), 5, allow_empty_subset=False))
    payload = valid.model_dump(mode="json")
    mutations = (
        {**payload, "source": {"items": ["3", "1", "5"]}},
        {**payload, "target": "4"},
        {
            **payload,
            "status": "NOT_ATTAINED",
            "witness": None,
            "reconstructed_sum": None,
        },
        {**payload, "witness": {"indices": [2]}},
        {**payload, "reconstructed_sum": "6"},
    )
    for mutation in mutations:
        with pytest.raises(ValidationError):
            SubsetSumTargetResult.model_validate(mutation)

    empty = _operation().run(_request((), 0, allow_empty_subset=True))
    with pytest.raises(ValidationError):
        SubsetSumTargetResult.model_validate(
            {**empty.model_dump(mode="json"), "allow_empty_subset": False}
        )


def test_result_bounds_raw_witness_and_sum_before_replay() -> None:
    valid = _operation().run(_request((3, 2, 5), 5, allow_empty_subset=False))
    payload = valid.model_dump(mode="json")

    with pytest.raises(ValidationError, match="outside its source"):
        SubsetSumTargetResult.model_validate({**payload, "witness": {"indices": [3]}})
    with pytest.raises(ValidationError, match="262-digit result bound"):
        SubsetSumTargetResult.model_validate(
            {**payload, "reconstructed_sum": "9" * 263}
        )


def test_request_admits_large_low_state_and_exact_digit_and_state_boundaries() -> None:
    zeros = _operation().run(_request((0,) * 256, 0, allow_empty_subset=False))
    assert zeros.witness == IndexSubset(indices=(0,))

    many_zeros = _operation().run(_request((0,) * 256, 0, allow_empty_subset=True))
    assert many_zeros.witness == IndexSubset(indices=())

    widest = "9" * 256
    wide_result = _operation().run(
        SubsetSumTargetRequest(
            source=IndexedIntegerSequence(items=(widest,)),
            target=widest,
            allow_empty_subset=False,
        )
    )
    assert wide_result.witness == IndexSubset(indices=(0,))

    powers = tuple(1 << exponent for exponent in range(16))
    state_boundary = _operation().run(
        _request(powers, sum(powers), allow_empty_subset=True)
    )
    assert state_boundary.witness == IndexSubset(indices=tuple(range(16)))


def test_request_rejects_immediately_above_each_search_bound() -> None:
    with pytest.raises(ValidationError, match="256-digit"):
        SubsetSumTargetRequest(
            source={"items": ["1" + "0" * 256]},
            target="0",
            allow_empty_subset=False,
        )
    with pytest.raises(ValidationError, match="262-digit"):
        SubsetSumTargetRequest(
            source={"items": ["-" + "9" * 256]},
            target="-" + "9" * 263,
            allow_empty_subset=True,
        )
    with pytest.raises(ValidationError, match="262-digit"):
        SubsetSumTargetRequest(
            source={"items": ["-" + "9" * 256]},
            target="9" * 263,
            allow_empty_subset=True,
        )

    widest = "9" * 256
    above_wire_count = (4 * 1024 * 1024 - 64) // (len(widest) + 4) + 1
    with pytest.raises(ValidationError, match="4 MiB wire-size"):
        SubsetSumTargetRequest(
            source={"items": [widest] * above_wire_count},
            target="0",
            allow_empty_subset=True,
        )

    with pytest.raises(ValidationError, match="complete-call bound"):
        SubsetSumTargetRequest.model_validate(
            {
                "source": {"items": ["not-an-integer"] * (500_000 + 1)},
                "target": "0",
                "allow_empty_subset": True,
            }
        )

    # Powers of two below 2**16 plus a 2**17 item leave the gap
    # (65,535, 131,072) unattainable: target 100000 exhausts the scan and
    # pushes 2**17 reachable states past the 65,536-state bound.
    with pytest.raises(ValidationError, match="65,536-reachable-state"):
        SubsetSumTargetRequest(
            source={
                "items": [str(1 << exponent) for exponent in range(16)] + ["131072"]
            },
            target="100000",
            allow_empty_subset=True,
        )

    with pytest.raises(ValidationError, match="2,000,000-transition complete-call"):
        _request((2,) * 1000, 3, allow_empty_subset=True)


def test_zero_targets_resolve_before_expansion_without_the_empty_subset() -> None:
    # With the empty subset inadmissible, every admissible subset of a
    # strictly positive source has positive sum, so target 0 is exactly
    # unattainable even though expanding 2**17 reachable states would exceed
    # the 65,536-state bound.
    powers = tuple(1 << exponent for exponent in range(17))
    positive = _operation().run(_request(powers, 0, allow_empty_subset=False))
    assert positive.status == "NOT_ATTAINED"
    assert positive.witness is None
    assert positive.reconstructed_sum is None

    replayed = SubsetSumTargetResult.model_validate_json(positive.model_dump_json())
    assert replayed == positive

    # Mirror image: every admissible subset of a strictly negative source has
    # negative sum, so target 0 is again exactly unattainable.
    negative = _operation().run(
        _request(tuple(-value for value in powers), 0, allow_empty_subset=False)
    )
    assert negative.status == "NOT_ATTAINED"
    assert negative.witness is None
    assert negative.reconstructed_sum is None

    # A zero item still attains zero without the empty subset.
    zero_item = _operation().run(_request((3, 0, 5), 0, allow_empty_subset=False))
    assert zero_item.status == "ATTAINED"
    assert zero_item.witness == IndexSubset(indices=(1,))
    assert zero_item.reconstructed_sum == "0"

    # The empty subset remains admissible here, so the same request resolves
    # to it instead of expanding.
    empty_witness = _operation().run(_request(powers, 0, allow_empty_subset=True))
    assert empty_witness.status == "ATTAINED"
    assert empty_witness.witness == IndexSubset(indices=())
    assert empty_witness.reconstructed_sum == "0"


def test_out_of_range_targets_resolve_before_state_expansion() -> None:
    # Every subset sum of a positive-only source is nonnegative, so -1 is
    # exactly unattainable even though an exhaustive expansion would push
    # 2**17 reachable states past the 65,536-state bound.
    powers = tuple(1 << exponent for exponent in range(17))
    below = _operation().run(_request(powers, -1, allow_empty_subset=True))
    assert below.status == "NOT_ATTAINED"
    assert below.witness is None
    assert below.reconstructed_sum is None

    replayed = SubsetSumTargetResult.model_validate_json(below.model_dump_json())
    assert replayed == below

    # Mirror image: every subset sum of a negative-only source is
    # nonpositive, so +1 is exactly unattainable without any expansion.
    mirrored = _operation().run(
        _request(tuple(-value for value in powers), 1, allow_empty_subset=True)
    )
    assert mirrored.status == "NOT_ATTAINED"
    assert mirrored.witness is None

    above = _operation().run(_request((1,) * 999, -1, allow_empty_subset=False))
    assert above.status == "NOT_ATTAINED"

    # Targets on the attained interval boundary stay inside the search:
    # the negative span itself is attained by selecting every negative item.
    negatives = (-2, -3, -7)
    edge = _operation().run(_request(negatives, -12, allow_empty_subset=False))
    assert edge.status == "ATTAINED"
    assert edge.witness == IndexSubset(indices=(0, 1, 2))

    just_inside = _operation().run(_request(negatives, -11, allow_empty_subset=True))
    assert just_inside.status == "NOT_ATTAINED"


def test_complete_call_charges_all_four_reachable_state_passes() -> None:
    assert (
        MAX_SUBSET_SUM_TOTAL_TRANSITIONS
        == MAX_SUBSET_SUM_COMPLETE_CALL_PASSES * MAX_SUBSET_SUM_TRANSITIONS_PER_PASS
    )

    # An in-range exhausting request whose single pass scans 499,500 states
    # stays admitted: its four charged passes (admission, kernel, and both
    # source-binding replays) fit the advertised complete-call budget.
    dense = (2,) * 999
    unattained = _operation().run(_request(dense, 3, allow_empty_subset=True))
    assert unattained.status == "NOT_ATTAINED"
    assert unattained.witness is None

    replayed = SubsetSumTargetResult.model_validate_json(unattained.model_dump_json())
    assert replayed == unattained


def test_request_admits_sources_at_the_profile_item_ceiling() -> None:
    wide_source = IndexedIntegerSequence(items=("0",) * MAX_SUBSET_SUM_ITEMS)
    assert len(wide_source.items) == MAX_SUBSET_SUM_ITEMS

    zeros = _operation().run(
        SubsetSumTargetRequest(
            source=wide_source,
            target="0",
            allow_empty_subset=False,
        )
    )
    assert zeros.status == "ATTAINED"
    assert zeros.witness == IndexSubset(indices=(0,))
    assert zeros.reconstructed_sum == "0"

    dense = 512
    ones = _operation().run(_request((1,) * dense, dense - 1, allow_empty_subset=True))
    assert ones.status == "ATTAINED"
    assert ones.witness == IndexSubset(indices=tuple(range(dense - 1)))
    assert ones.reconstructed_sum == str(dense - 1)


def test_request_accepts_targets_at_the_derived_subset_sum_width() -> None:
    widest = "9" * 256
    attained_total = str(2 * int(widest))
    assert len(attained_total) == 257

    attained = _operation().run(
        SubsetSumTargetRequest(
            source=IndexedIntegerSequence(items=(widest, widest)),
            target=attained_total,
            allow_empty_subset=False,
        )
    )
    assert attained.status == "ATTAINED"
    assert attained.witness == IndexSubset(indices=(0, 1))
    assert attained.reconstructed_sum == attained_total

    negative_width = _operation().run(
        SubsetSumTargetRequest(
            source=IndexedIntegerSequence(items=(widest, widest)),
            target="-" + attained_total,
            allow_empty_subset=True,
        )
    )
    assert negative_width.status == "NOT_ATTAINED"

    same_width = _operation().run(
        SubsetSumTargetRequest(
            source=IndexedIntegerSequence(items=("5", "5")),
            target="99",
            allow_empty_subset=True,
        )
    )
    assert same_width.status == "NOT_ATTAINED"


def test_request_resolves_targets_beyond_the_derived_subset_sum_width() -> None:
    # Source (5,) attains [0, 5]; target 99 lies outside with more digits
    # than either endpoint, so admission resolves it exactly without any
    # source-sensitive width restriction or state expansion.
    beyond = _operation().run(_request((5,), 99, allow_empty_subset=True))
    assert beyond.status == "NOT_ATTAINED"
    assert beyond.witness is None
    assert beyond.reconstructed_sum is None

    replayed = SubsetSumTargetResult.model_validate_json(beyond.model_dump_json())
    assert replayed == beyond

    empty = _operation().run(_request((), 10, allow_empty_subset=True))
    assert empty.status == "NOT_ATTAINED"

    wide = _operation().run(
        SubsetSumTargetRequest(
            source={"items": []},
            target="1" + "0" * 256,
            allow_empty_subset=False,
        )
    )
    assert wide.status == "NOT_ATTAINED"


def test_resolved_empty_witness_skips_state_expansion_admission() -> None:
    resolved = _operation().run(
        _request(
            tuple(1 << exponent for exponent in range(20)),
            0,
            allow_empty_subset=True,
        )
    )
    assert resolved.status == "ATTAINED"
    assert resolved.witness == IndexSubset(indices=())
    assert resolved.reconstructed_sum == "0"

    replayed = SubsetSumTargetResult.model_validate_json(resolved.model_dump_json())
    assert replayed == resolved

    saturated = _operation().run(
        _request((0,) * MAX_SUBSET_SUM_ITEMS, 0, allow_empty_subset=True)
    )
    assert saturated.status == "ATTAINED"
    assert saturated.witness == IndexSubset(indices=())


def test_request_admits_targets_attained_by_an_early_source_prefix() -> None:
    powers = tuple(1 << exponent for exponent in range(20))

    singleton = _operation().run(_request(powers, 1, allow_empty_subset=False))
    assert singleton.status == "ATTAINED"
    assert singleton.witness == IndexSubset(indices=(0,))
    assert singleton.reconstructed_sum == "1"

    second_item = _operation().run(_request(powers, 2, allow_empty_subset=True))
    assert second_item.witness == IndexSubset(indices=(1,))

    zero_singleton = _operation().run(
        _request(
            tuple(1 << exponent for exponent in range(16)) + (0,) * 7,
            0,
            allow_empty_subset=False,
        )
    )
    assert zero_singleton.witness == IndexSubset(indices=(16,))

    replayed = SubsetSumTargetResult.model_validate_json(
        singleton.model_dump_json(),
        strict=True,
    )
    assert replayed == singleton


def test_request_admits_sources_beyond_the_profile_item_ceiling() -> None:
    beyond_profile = MAX_SUBSET_SUM_ITEMS + 1
    assert beyond_profile > MAX_SUBSET_SUM_ITEMS

    zeros = _operation().run(
        SubsetSumTargetRequest(
            source=IndexedIntegerSequence(items=("0",) * beyond_profile),
            target="0",
            allow_empty_subset=False,
        )
    )
    assert zeros.status == "ATTAINED"
    assert zeros.witness == IndexSubset(indices=(0,))
    assert zeros.reconstructed_sum == "0"

    replayed = SubsetSumTargetResult.model_validate_json(zeros.model_dump_json())
    assert replayed == zeros


def test_request_admits_low_state_sources_with_wide_lattice_spans() -> None:
    huge = "1" + "0" * 100
    wide = _operation().run(
        SubsetSumTargetRequest(
            source=IndexedIntegerSequence(items=(huge, "-" + huge)),
            target="5",
            allow_empty_subset=False,
        )
    )
    assert wide.status == "NOT_ATTAINED"
    assert wide.witness is None


def test_resolving_scan_is_charged_against_the_transition_bound() -> None:
    powers = tuple(1 << exponent for exponent in range(16))
    target = str(1 << 16)

    admitted = _operation().run(
        _request(powers + (0,) * 5 + (1,), int(target), allow_empty_subset=True)
    )
    assert admitted.status == "ATTAINED"
    assert admitted.witness == IndexSubset(indices=(*range(16), 21))
    assert admitted.reconstructed_sum == target

    replayed = SubsetSumTargetResult.model_validate_json(admitted.model_dump_json())
    assert replayed == admitted

    with pytest.raises(ValidationError, match="complete-call"):
        SubsetSumTargetRequest(
            source={"items": [str(value) for value in powers + (0,) * 6 + (1,)]},
            target=target,
            allow_empty_subset=True,
        )


def test_target_sources_compose_across_indexed_sequence_operations() -> None:
    sequence = IndexedIntegerSequence(items=("2", "3"))
    attained = _operation().run(
        SubsetSumTargetRequest(source=sequence, target="5", allow_empty_subset=False)
    )
    assert type(attained.source) is IndexedIntegerSequence
    assert attained.source == sequence

    profile = subset_sum_profile(attained.source)
    assert profile.source == sequence

    residue = compute_subset_sum_residue_profile(
        SubsetSumResidueProfileRequest(
            source=attained.source,
            modulus=5,
            include_empty_subset=False,
            include_witnesses=False,
        )
    )
    assert residue.source == sequence


def test_schema_publishes_enforced_target_and_source_bounds() -> None:
    schema = SubsetSumTargetRequest.model_json_schema()

    target_schema = schema["properties"]["target"]
    assert target_schema["maxLength"] == MAX_SUBSET_SUM_RECONSTRUCTED_DIGITS + 1
    assert target_schema["pattern"] == (
        rf"^(?:0|-?[1-9][0-9]{{0,{MAX_SUBSET_SUM_RECONSTRUCTED_DIGITS - 1}}})$"
    )

    items_schema = schema["properties"]["source"]["properties"]["items"]
    assert items_schema["maxItems"] == 500_000
    description = items_schema["description"]
    assert f"{MAX_SUBSET_SUM_INTEGER_DIGITS} decimal digits" in description
    item_schema = items_schema["items"]
    assert item_schema["maxLength"] == MAX_SUBSET_SUM_INTEGER_DIGITS + 1
    assert item_schema["pattern"] == (
        rf"^(?:0|-?[1-9][0-9]{{0,{MAX_SUBSET_SUM_INTEGER_DIGITS - 1}}})$"
    )
    assert "IndexedIntegerSequence" not in schema.get("$defs", {})

    result_schema = SubsetSumTargetResult.model_json_schema()
    result_item_schema = result_schema["properties"]["source"]["properties"]["items"][
        "items"
    ]
    assert result_item_schema["maxLength"] == MAX_SUBSET_SUM_INTEGER_DIGITS + 1
    assert result_item_schema["pattern"] == item_schema["pattern"]


def test_published_item_pattern_matches_the_enforced_digit_boundary() -> None:
    schema = SubsetSumTargetRequest.model_json_schema()
    pattern = re.compile(
        schema["properties"]["source"]["properties"]["items"]["items"]["pattern"]
    )

    at_ceiling = "9" * MAX_SUBSET_SUM_INTEGER_DIGITS
    beyond = "1" + "0" * MAX_SUBSET_SUM_INTEGER_DIGITS
    assert pattern.fullmatch(at_ceiling)
    assert not pattern.fullmatch(beyond)

    # The same 257-digit item the published pattern rejects is rejected by
    # typed request validation with the enforced bound named.
    with pytest.raises(ValidationError, match="256-digit"):
        SubsetSumTargetRequest.model_validate(
            {
                "source": {"items": [beyond]},
                "target": "0",
                "allow_empty_subset": True,
            }
        )


def test_target_scalar_pattern_encodes_the_absolute_digit_ceiling() -> None:
    adapter = TypeAdapter(_SubsetSumTargetScalar)

    assert adapter.validate_python("9" * 262) == "9" * 262
    # Only a negative 262-digit value needs the extra sign character.
    assert adapter.validate_python("-" + "9" * 262) == "-" + "9" * 262

    with pytest.raises(ValidationError):
        adapter.validate_python("9" * 263)
    with pytest.raises(ValidationError):
        adapter.validate_python("-" + "9" * 263)
    with pytest.raises(ValidationError):
        adapter.validate_python("0" * 262)
    with pytest.raises(ValidationError):
        adapter.validate_python("0" + "9" * 261)
