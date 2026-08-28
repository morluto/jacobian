"""Contract tests for complete canonical Nim option families."""

from __future__ import annotations

from copy import deepcopy

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.math.logic.games.impartial._models import (
    NimOptionsRequest,
    NimOptionsResult,
    NimSumRequest,
)
from jacobian.math.logic.games.impartial._operations import (
    compute_nim_options,
    compute_nim_sum,
)
from jacobian.math.logic.games.impartial._tools import TOOLS
from jacobian.math.logic.games.impartial.values import (
    MAX_NIM_DISTINCT_OPTIONS,
    MAX_NIM_OPTION_RESULT_BYTES,
    MAX_NIM_RAW_CANDIDATES,
    NimPosition,
)


def test_nim_options_deduplicate_equal_heaps_and_retain_all_source_indices() -> None:
    result = compute_nim_options(
        NimOptionsRequest(position=NimPosition(heaps=(1, 1, 2)))
    )

    assert tuple(
        (
            option.resulting_position.heaps,
            option.source_heap_indices,
            option.source_heap_size,
            option.replacement_heap_size,
        )
        for option in result.options
    ) == (
        ((0, 1, 1), (2,), 2, 0),
        ((0, 1, 2), (0, 1), 1, 0),
        ((1, 1, 1), (2,), 2, 1),
    )
    assert result.raw_candidate_count == 4
    assert result.distinct_option_count == 3


@pytest.mark.parametrize("heaps", ((), (0,), (0, 0, 0)))
def test_terminal_nim_positions_have_one_complete_empty_option_family(
    heaps: tuple[int, ...],
) -> None:
    result = compute_nim_options(NimOptionsRequest(position=NimPosition(heaps=heaps)))

    assert result.options == ()
    assert result.raw_candidate_count == 0
    assert result.distinct_option_count == 0


@given(
    st.lists(
        st.integers(min_value=0, max_value=5),
        min_size=0,
        max_size=6,
    ).map(lambda heaps: tuple(sorted(heaps)))
)
def test_nim_options_match_direct_indexed_move_enumeration(
    heaps: tuple[int, ...],
) -> None:
    result = compute_nim_options(NimOptionsRequest(position=NimPosition(heaps=heaps)))
    indexed_options: dict[tuple[int, ...], set[int]] = {}
    for index, heap in enumerate(heaps):
        for replacement in range(heap):
            option = list(heaps)
            option[index] = replacement
            canonical = tuple(sorted(option))
            indexed_options.setdefault(canonical, set()).add(index)

    expected = tuple(
        (position, tuple(sorted(indices)))
        for position, indices in sorted(indexed_options.items())
    )
    actual = tuple(
        (option.resulting_position.heaps, option.source_heap_indices)
        for option in result.options
    )

    assert actual == expected
    assert result.raw_candidate_count == sum(heaps)
    assert result.distinct_option_count == len(indexed_options)


def test_nim_position_contract_is_canonical_strict_and_bounded() -> None:
    with pytest.raises(ValidationError):
        NimPosition(heaps=(2, 1))
    with pytest.raises(ValidationError):
        NimPosition.model_validate({"heaps": ["1"]})
    with pytest.raises(ValidationError):
        NimPosition.model_validate({"heaps": [1.0]})
    with pytest.raises(ValidationError):
        NimPosition(heaps=(10_001,))
    with pytest.raises(ValidationError):
        NimPosition(heaps=(0,) * 51)


def test_nim_options_preflight_distinguishes_raw_and_distinct_counts() -> None:
    max_raw = NimOptionsRequest(position=NimPosition(heaps=(10_000,) * 50))
    assert max_raw.position.heaps == (10_000,) * 50
    assert sum(max_raw.position.heaps) == MAX_NIM_RAW_CANDIDATES

    exact_distinct = (15, 9_995, 9_996, 9_997, 9_998, 9_999)
    assert sum(set(exact_distinct)) == MAX_NIM_DISTINCT_OPTIONS
    NimOptionsRequest(position=NimPosition(heaps=exact_distinct))

    with pytest.raises(ValueError):
        compute_nim_options(
            NimOptionsRequest(
                position=NimPosition(heaps=(16, 9_995, 9_996, 9_997, 9_998, 9_999))
            )
        )


def test_nim_options_reject_result_bytes_before_option_expansion() -> None:
    with pytest.raises(ValueError):
        compute_nim_options(
            NimOptionsRequest(position=NimPosition(heaps=tuple(range(951, 1_001))))
        )


def test_nim_options_result_is_source_bound_and_canonical_json_bounded() -> None:
    result = compute_nim_options(
        NimOptionsRequest(position=NimPosition(heaps=(1, 2, 2)))
    )
    payload = result.model_dump(mode="json")

    assert len(encode_strict_json(payload)) <= MAX_NIM_OPTION_RESULT_BYTES
    assert (
        NimOptionsRequest.model_validate(
            {"position": result.options[0].resulting_position.model_dump(mode="json")}
        ).position
        == result.options[0].resulting_position
    )
    downstream_request = NimSumRequest.model_validate(
        {"position": result.options[0].resulting_position.model_dump(mode="json")}
    )
    downstream = compute_nim_sum(downstream_request)
    assert downstream.position == result.options[0].resulting_position

    order_mutation = deepcopy(payload)
    order_mutation["options"][0], order_mutation["options"][1] = (
        order_mutation["options"][1],
        order_mutation["options"][0],
    )
    with pytest.raises(ValidationError):
        NimOptionsResult.model_validate(order_mutation)

    index_bound_mutation = deepcopy(payload)
    index_bound_mutation["options"][0]["source_heap_indices"] = [50]
    with pytest.raises(ValidationError):
        NimOptionsResult.model_validate(index_bound_mutation)


def test_nim_options_is_a_public_exact_operation() -> None:
    tool = next(
        tool for tool in TOOLS if tool.operation_id == "game.nim.options.compute"
    )
    request = tool.request_type.model_validate(tool.examples[0].input)
    result = tool.run(request)

    assert isinstance(result, NimOptionsResult)
    assert result.distinct_option_count == len(result.options)
    assert {"game-theory", "nim", "options", "exact"} <= set(tool.tags)
