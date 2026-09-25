"""Complete LR tableau enumeration checked by a direct small-shape oracle."""

from __future__ import annotations

from itertools import permutations

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.symmetric_functions import (
    IntegerPartition,
    littlewood_richardson_coefficient,
    littlewood_richardson_tableaux,
)
from jacobian.math.combinatorics.symmetric_functions._models import (
    LittlewoodRichardsonTableauxRequest,
)
from jacobian.math.combinatorics.symmetric_functions._tools import TOOLS


def _partitions(total: int, maximum: int | None = None) -> tuple[tuple[int, ...], ...]:
    if total == 0:
        return ((),)
    limit = total if maximum is None else min(total, maximum)
    return tuple(
        (first, *rest)
        for first in range(limit, 0, -1)
        for rest in _partitions(total - first, first)
    )


def _is_contained(inner: tuple[int, ...], outer: tuple[int, ...]) -> bool:
    return all(
        value <= (outer[index] if index < len(outer) else 0)
        for index, value in enumerate(inner)
    )


def _brute_force_family(
    outer: tuple[int, ...], inner: tuple[int, ...], content: tuple[int, ...]
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Try every content word and check the defining tableau conditions."""
    if not _is_contained(inner, outer):
        return ()
    skew_size = sum(outer) - sum(inner)
    if skew_size != sum(content):
        return ()
    cells = tuple(
        (row, column)
        for row, width in enumerate(outer)
        for column in range(width, inner[row] if row < len(inner) else 0, -1)
    )
    multiset = tuple(
        value
        for value, multiplicity in enumerate(content, start=1)
        for _ in range(multiplicity)
    )
    valid: list[tuple[tuple[int, ...], ...]] = []
    for word in set(permutations(multiset)):
        counts = [0] * len(content)
        lattice = True
        for value in word:
            counts[value - 1] += 1
            if any(
                counts[index] < counts[index + 1] for index in range(len(counts) - 1)
            ):
                lattice = False
                break
        if not lattice:
            continue
        values = dict(zip(cells, word, strict=True))
        if any(
            values[(row, column)] > values[(row, column + 1)]
            for row, width in enumerate(outer)
            for column in range(
                (inner[row] if row < len(inner) else 0) + 1,
                width,
            )
        ):
            continue
        if any(
            values[(upper, column)] >= values[(lower, column)]
            for lower, width in enumerate(outer)
            for upper in range(lower)
            for column in range(1, width + 1)
            if (upper, column) in values and (lower, column) in values
        ):
            continue
        rows = tuple(
            tuple(
                values[(row, column)]
                for column in range(
                    (inner[row] if row < len(inner) else 0) + 1,
                    width + 1,
                )
            )
            for row, width in enumerate(outer)
            if width > (inner[row] if row < len(inner) else 0)
        )
        valid.append(rows)
    return tuple(
        sorted(
            valid,
            key=lambda rows: tuple(value for row in rows for value in reversed(row)),
        )
    )


@pytest.mark.parametrize("size", range(5))
def test_lr_tableau_families_match_exhaustive_small_filling_oracle(size: int) -> None:
    for outer in _partitions(size):
        for inner_size in range(size + 1):
            for inner in _partitions(inner_size):
                if not _is_contained(inner, outer):
                    continue
                skew_size = size - inner_size
                for content in _partitions(skew_size):
                    request = tuple(
                        IntegerPartition(parts=parts)
                        for parts in (outer, inner, content)
                    )
                    result = littlewood_richardson_tableaux(*request)
                    observed = tuple(tableau.rows for tableau in result.tableaux)
                    expected = _brute_force_family(outer, inner, content)
                    assert observed == expected
                    coefficient = littlewood_richardson_coefficient(*request)
                    assert coefficient.coefficient == len(observed)
                    assert len(observed) == len(set(observed))


def test_empty_skew_shape_has_one_empty_lr_tableau() -> None:
    empty = IntegerPartition(parts=())
    shape = IntegerPartition(parts=(3, 1))
    result = littlewood_richardson_tableaux(shape, shape, empty)
    assert tuple(tableau.rows for tableau in result.tableaux) == ((),)


def test_impossible_large_shape_returns_empty_without_search_admission() -> None:
    outer = IntegerPartition(parts=(500,))
    empty = IntegerPartition(parts=())
    result = littlewood_richardson_tableaux(outer, empty, empty)
    assert result.tableaux == ()


def test_lr_tableau_enumeration_rejects_before_expansion_and_is_catalogued() -> None:
    with pytest.raises(ValidationError, match="lr_skew_size_exceeded"):
        LittlewoodRichardsonTableauxRequest(
            outer=IntegerPartition(parts=(9,)),
            inner=IntegerPartition(parts=()),
            content=IntegerPartition(parts=(9,)),
        )
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "combinatorics.littlewood_richardson.tableaux.enumerate"
    )
    assert tool.examples[0].input["outer"] == {"parts": [3, 2, 1]}
