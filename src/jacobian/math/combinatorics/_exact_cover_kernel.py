"""Deterministic bounded Algorithm X kernel for generalized exact cover."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from jacobian._execution import report_request_progress, request_checkpoint
from jacobian.math.combinatorics.exact_cover import GeneralizedExactCoverInstance


@dataclass(frozen=True, slots=True)
class _SearchState:
    uncovered_primary: int
    available_rows: int
    selected_rows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExactCoverKernelResult:
    status: Literal["FOUND", "NO_COVER", "UNKNOWN"]
    selected_rows: tuple[int, ...] = ()
    frontier_prefixes: tuple[tuple[int, ...], ...] = ()
    visited_nodes: int = 0


def _indices(
    instance: GeneralizedExactCoverInstance,
) -> tuple[list[int], list[int], list[int]]:
    items = (*instance.primary_items, *instance.secondary_items)
    item_index = {item: index for index, item in enumerate(items)}
    primary_count = len(instance.primary_items)
    item_rows = [0] * len(items)
    row_item_indices: list[tuple[int, ...]] = []
    row_primary_masks: list[int] = []
    for row_index, row in enumerate(instance.rows):
        indices = tuple(item_index[item] for item in row.items)
        row_item_indices.append(indices)
        primary_mask = 0
        for index in indices:
            item_rows[index] |= 1 << row_index
            if index < primary_count:
                primary_mask |= 1 << index
        row_primary_masks.append(primary_mask)
    row_conflicts = []
    for indices in row_item_indices:
        conflicts = 0
        for index in indices:
            conflicts |= item_rows[index]
        row_conflicts.append(conflicts)
    return item_rows, row_primary_masks, row_conflicts


def _children(
    state: _SearchState,
    item_rows: list[int],
    row_primary_masks: list[int],
    row_conflicts: list[int],
) -> tuple[_SearchState, ...]:
    chosen_rows = 0
    fewest = len(row_primary_masks) + 1
    remaining = state.uncovered_primary
    while remaining:
        bit = remaining & -remaining
        item = bit.bit_length() - 1
        candidates = item_rows[item] & state.available_rows
        if candidates.bit_count() < fewest:
            fewest = candidates.bit_count()
            chosen_rows = candidates
        remaining ^= bit
    children = []
    while chosen_rows:
        bit = chosen_rows & -chosen_rows
        row = bit.bit_length() - 1
        children.append(
            _SearchState(
                uncovered_primary=state.uncovered_primary & ~row_primary_masks[row],
                available_rows=state.available_rows & ~row_conflicts[row],
                selected_rows=(*state.selected_rows, row),
            )
        )
        chosen_rows ^= bit
    return tuple(children)


def search_generalized_exact_cover(
    instance: GeneralizedExactCoverInstance,
    search_node_limit: int,
    fixed_rows: tuple[int, ...] = (),
) -> ExactCoverKernelResult:
    """Search a deterministic prefix and retain every unresolved subtree."""

    if search_node_limit < 1:
        raise ValueError("search_node_limit must be positive")
    item_rows, row_primary_masks, row_conflicts = _indices(instance)
    state = _SearchState(
        uncovered_primary=(1 << len(instance.primary_items)) - 1,
        available_rows=(1 << len(instance.rows)) - 1,
        selected_rows=(),
    )
    for fixed_row in fixed_rows:
        matches = [
            child
            for child in _children(state, item_rows, row_primary_masks, row_conflicts)
            if child.selected_rows[-1] == fixed_row
        ]
        if not matches:
            raise ValueError("fixed-row prefix is not a semantic traversal prefix")
        state = matches[0]
    stack = [state]
    visited = 0
    while stack and visited < search_node_limit:
        state = stack.pop()
        visited += 1
        if visited == 1 or visited % 256 == 0:
            request_checkpoint("during exact-cover search")
            report_request_progress(visited, message="exact-cover search nodes visited")
        if state.uncovered_primary == 0:
            return ExactCoverKernelResult(
                status="FOUND",
                selected_rows=state.selected_rows,
                visited_nodes=visited,
            )
        children = _children(state, item_rows, row_primary_masks, row_conflicts)
        stack.extend(reversed(children))
    if stack:
        return ExactCoverKernelResult(
            status="UNKNOWN",
            frontier_prefixes=tuple(state.selected_rows for state in reversed(stack)),
            visited_nodes=visited,
        )
    return ExactCoverKernelResult(status="NO_COVER", visited_nodes=visited)


def split_exact_cover_prefix(
    instance: GeneralizedExactCoverInstance,
    fixed_rows: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Split one semantic traversal prefix into disjoint complete children."""

    item_rows, row_primary_masks, row_conflicts = _indices(instance)
    state = _SearchState(
        uncovered_primary=(1 << len(instance.primary_items)) - 1,
        available_rows=(1 << len(instance.rows)) - 1,
        selected_rows=(),
    )
    for fixed_row in fixed_rows:
        children = _children(state, item_rows, row_primary_masks, row_conflicts)
        matches = [child for child in children if child.selected_rows[-1] == fixed_row]
        if not matches:
            raise ValueError("fixed-row prefix is not a semantic traversal prefix")
        state = matches[0]
    if state.uncovered_primary == 0:
        raise ValueError("a completed cover cannot be split as an unresolved shard")
    return tuple(
        child.selected_rows
        for child in _children(state, item_rows, row_primary_masks, row_conflicts)
    )


__all__ = [
    "ExactCoverKernelResult",
    "search_generalized_exact_cover",
    "split_exact_cover_prefix",
]
