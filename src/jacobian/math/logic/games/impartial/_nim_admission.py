"""Nim-option admission quantities shared by the request and native kernel."""

from __future__ import annotations

from jacobian.math.logic.games.impartial.values import (
    MAX_NIM_DISTINCT_OPTIONS,
    MAX_NIM_RAW_CANDIDATES,
    NimPosition,
)


def heap_groups(position: NimPosition) -> tuple[tuple[int, tuple[int, ...]], ...]:
    groups: dict[int, list[int]] = {}
    for index, heap in enumerate(position.heaps):
        groups.setdefault(heap, []).append(index)
    return tuple((heap, tuple(indices)) for heap, indices in groups.items())


def admit_nim_options(
    position: NimPosition,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Bound complete indexed Nim options and return prepared heap groups."""

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
    return groups


__all__ = ["admit_nim_options", "heap_groups"]
