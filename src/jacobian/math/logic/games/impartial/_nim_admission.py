"""Nim-option admission quantities shared by the request and native kernel."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.canonical import strict_json_object_size
from jacobian.math.logic.games.impartial.values import (
    MAX_NIM_DISTINCT_OPTIONS,
    MAX_NIM_OPTION_RESULT_BYTES,
    MAX_NIM_RAW_CANDIDATES,
    NimPosition,
)


@dataclass(frozen=True, slots=True)
class NimOptionPlan:
    """The exact bounded shape of one complete Nim option result."""

    raw_candidate_count: int
    distinct_option_count: int
    serialized_result_bytes: int


def heap_groups(position: NimPosition) -> tuple[tuple[int, tuple[int, ...]], ...]:
    groups: dict[int, list[int]] = {}
    for index, heap in enumerate(position.heaps):
        groups.setdefault(heap, []).append(index)
    return tuple((heap, tuple(indices)) for heap, indices in groups.items())


def _json_integer_size(value: int) -> int:
    return len(str(value))


def _json_integer_sequence_size(values: tuple[int, ...]) -> int:
    return (
        2 + max(len(values) - 1, 0) + sum(_json_integer_size(value) for value in values)
    )


def _nim_position_json_size(heap_list_size: int) -> int:
    return strict_json_object_size((("heaps", heap_list_size),))


def _nim_option_result_size(
    position: NimPosition,
    groups: tuple[tuple[int, tuple[int, ...]], ...],
    distinct_option_count: int,
    raw_candidate_count: int,
) -> int:
    source_heap_list_size = _json_integer_sequence_size(position.heaps)
    row_sizes = 0
    for source_size, source_indices in groups:
        if source_size == 0:
            continue
        source_index_size = _json_integer_sequence_size(source_indices)
        for replacement_size in range(source_size):
            resulting_heap_list_size = (
                source_heap_list_size
                - _json_integer_size(source_size)
                + _json_integer_size(replacement_size)
            )
            row_sizes += strict_json_object_size(
                (
                    ("source_heap_indices", source_index_size),
                    ("source_heap_size", _json_integer_size(source_size)),
                    ("replacement_heap_size", _json_integer_size(replacement_size)),
                    (
                        "resulting_position",
                        _nim_position_json_size(resulting_heap_list_size),
                    ),
                )
            )
    options_size = 2 + max(distinct_option_count - 1, 0) + row_sizes
    return strict_json_object_size(
        (
            ("position", _nim_position_json_size(source_heap_list_size)),
            ("options", options_size),
            ("raw_candidate_count", _json_integer_size(raw_candidate_count)),
            ("distinct_option_count", _json_integer_size(distinct_option_count)),
            ("complete", 4),
        )
    )


def nim_option_plan(position: NimPosition) -> NimOptionPlan:
    """Bound complete indexed Nim options before materialization."""

    groups = heap_groups(position)
    raw_candidate_count = sum(position.heaps)
    distinct_option_count = sum(heap for heap, _ in groups)
    if raw_candidate_count > MAX_NIM_RAW_CANDIDATES:
        raise ValueError(
            "indexed Nim moves exceed the exact raw-candidate bound of "
            f"{MAX_NIM_RAW_CANDIDATES}"
        )
    if distinct_option_count > MAX_NIM_DISTINCT_OPTIONS:
        raise ValueError(
            "distinct Nim options exceed the exact result-count bound of "
            f"{MAX_NIM_DISTINCT_OPTIONS}"
        )
    serialized_result_bytes = _nim_option_result_size(
        position, groups, distinct_option_count, raw_candidate_count
    )
    if serialized_result_bytes > MAX_NIM_OPTION_RESULT_BYTES:
        raise ValueError(
            "canonical serialized result exceeds the Nim option bound of "
            f"{MAX_NIM_OPTION_RESULT_BYTES} bytes"
        )
    return NimOptionPlan(
        raw_candidate_count=raw_candidate_count,
        distinct_option_count=distinct_option_count,
        serialized_result_bytes=serialized_result_bytes,
    )


__all__ = ["NimOptionPlan", "heap_groups", "nim_option_plan"]
