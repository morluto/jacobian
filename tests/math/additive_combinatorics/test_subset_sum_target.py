from __future__ import annotations

from itertools import permutations, product
from typing import cast

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import MathTool
from jacobian.math.additive_combinatorics import IndexSubset
from jacobian.math.additive_combinatorics._subset_sum_target import (
    SubsetSumTargetRequest,
    SubsetSumTargetResult,
    SubsetSumTargetSource,
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
        source=SubsetSumTargetSource(items=tuple(str(value) for value in values)),
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

    assert request.source == SubsetSumTargetSource(items=("2", "3"))
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
        source=SubsetSumTargetSource(items=("2", "3")),
        target="5",
        allow_empty_subset=False,
        status="ATTAINED",
        witness=IndexSubset(indices=(0, 1)),
        reconstructed_sum="5",
    )

    not_attained = _operation().run(_request((2, 3), 4, allow_empty_subset=False))
    assert not_attained == SubsetSumTargetResult(
        source=SubsetSumTargetSource(items=("2", "3")),
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
            source=SubsetSumTargetSource(items=(widest,)),
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
    with pytest.raises(ValidationError, match="attainable"):
        SubsetSumTargetRequest(
            source={"items": []},
            target="1" + "0" * 256,
            allow_empty_subset=False,
        )
    with pytest.raises(ValidationError, match="262-digit"):
        SubsetSumTargetRequest(
            source={"items": ["-" + "9" * 256]},
            target="-" + "9" * 263,
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

    with pytest.raises(ValidationError, match="65,536-reachable-state"):
        _request(
            tuple(1 << exponent for exponent in range(17)),
            0,
            allow_empty_subset=False,
        )

    with pytest.raises(ValidationError, match="1,000,000-transition complete-call"):
        _request((1,) * 1001, -1, allow_empty_subset=True)


def test_request_admits_sources_at_the_shared_item_ceiling() -> None:
    wide_source = SubsetSumTargetSource(items=("0",) * MAX_SUBSET_SUM_ITEMS)
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
            source=SubsetSumTargetSource(items=(widest, widest)),
            target=attained_total,
            allow_empty_subset=False,
        )
    )
    assert attained.status == "ATTAINED"
    assert attained.witness == IndexSubset(indices=(0, 1))
    assert attained.reconstructed_sum == attained_total

    negative_width = _operation().run(
        SubsetSumTargetRequest(
            source=SubsetSumTargetSource(items=(widest, widest)),
            target="-" + attained_total,
            allow_empty_subset=True,
        )
    )
    assert negative_width.status == "NOT_ATTAINED"

    same_width = _operation().run(
        SubsetSumTargetRequest(
            source=SubsetSumTargetSource(items=("5", "5")),
            target="99",
            allow_empty_subset=True,
        )
    )
    assert same_width.status == "NOT_ATTAINED"


def test_request_rejects_targets_beyond_the_derived_subset_sum_width() -> None:
    with pytest.raises(ValidationError, match="attainable"):
        SubsetSumTargetRequest(
            source={"items": ["5"]},
            target="99",
            allow_empty_subset=True,
        )

    with pytest.raises(ValidationError, match="1-digit attainable"):
        SubsetSumTargetRequest(
            source={"items": []},
            target="10",
            allow_empty_subset=True,
        )


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
            source=SubsetSumTargetSource(items=("0",) * beyond_profile),
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
            source=SubsetSumTargetSource(items=(huge, "-" + huge)),
            target="5",
            allow_empty_subset=False,
        )
    )
    assert wide.status == "NOT_ATTAINED"
    assert wide.witness is None
