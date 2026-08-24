"""Deterministic bounded Algorithm X kernel for generalized exact cover."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from jacobian.math.combinatorics.exact_cover import GeneralizedExactCoverInstance


@dataclass(frozen=True, slots=True)
class ExactCoverKernelResult:
    status: Literal["FOUND", "NO_COVER", "UNKNOWN"]
    selected_rows: tuple[int, ...] = ()


def search_generalized_exact_cover(
    instance: GeneralizedExactCoverInstance,
    search_node_limit: int,
) -> ExactCoverKernelResult:
    """Search one materialized primary/secondary incidence matrix.

    A node is one conflict-free partial selected-row family, represented by
    its uncovered-primary and available-row bitsets. The root and terminal
    states count as nodes. Branching chooses the uncovered primary item with
    the fewest available rows, ties by canonical item order, and tries rows in
    canonical row-ID order. Thus the same canonical input and node limit always
    produce the same status and witness.

    Write P for primary items, M for all items, R for rows, I for incidences,
    and N for the node allowance. Index construction materializes I item
    indices, M R-bit item-to-row masks, R P-bit row-to-primary masks, and R
    R-bit conflict masks using at most 2I mask updates. The logical bitset
    payload is therefore M*R + R*P + R*R bits plus I bounded item indices.

    Each node uses at most P intersections and population counts on R-bit
    masks to choose a primary item, followed by constant-many P- or R-bit mask
    operations for each child. At most N nodes and N-1 child edges are visited.
    Recursion depth is at most P, so its logical mask payload is bounded by
    (P+1)*(P+R) bits plus P selected row indices. The public P <= 256 and
    R <= 4096 bounds keep this below Python's recursion limit and make every
    intermediate integer width explicit.
    """

    if search_node_limit < 1:
        raise ValueError("search_node_limit must be positive")

    items = (*instance.primary_items, *instance.secondary_items)
    item_index = {item: index for index, item in enumerate(items)}
    primary_count = len(instance.primary_items)
    row_count = len(instance.rows)

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

    # Selecting a row removes every row sharing any primary or secondary item.
    # At most MAX_INCIDENCES bounded big-integer ORs construct this index.
    row_conflicts: list[int] = []
    for indices in row_item_indices:
        conflicts = 0
        for index in indices:
            conflicts |= item_rows[index]
        row_conflicts.append(conflicts)

    all_primary = (1 << primary_count) - 1
    all_rows = (1 << row_count) - 1
    visited_nodes = 0
    selected_rows: list[int] = []

    def visit(
        uncovered_primary: int,
        available_rows: int,
    ) -> ExactCoverKernelResult:
        nonlocal visited_nodes
        if visited_nodes >= search_node_limit:
            return ExactCoverKernelResult(status="UNKNOWN")
        visited_nodes += 1

        if uncovered_primary == 0:
            return ExactCoverKernelResult(
                status="FOUND", selected_rows=tuple(selected_rows)
            )

        chosen_rows = 0
        fewest_candidates = row_count + 1
        remaining_items = uncovered_primary
        while remaining_items:
            item_bit = remaining_items & -remaining_items
            item = item_bit.bit_length() - 1
            candidates = item_rows[item] & available_rows
            candidate_count = candidates.bit_count()
            if candidate_count < fewest_candidates:
                chosen_rows = candidates
                fewest_candidates = candidate_count
                if candidate_count == 0:
                    return ExactCoverKernelResult(status="NO_COVER")
            remaining_items ^= item_bit

        candidates = chosen_rows
        while candidates:
            row_bit = candidates & -candidates
            row = row_bit.bit_length() - 1
            selected_rows.append(row)
            result = visit(
                uncovered_primary & ~row_primary_masks[row],
                available_rows & ~row_conflicts[row],
            )
            selected_rows.pop()
            if result.status != "NO_COVER":
                return result
            candidates ^= row_bit
        return ExactCoverKernelResult(status="NO_COVER")

    return visit(all_primary, all_rows)


__all__ = ["ExactCoverKernelResult", "search_generalized_exact_cover"]
