from __future__ import annotations

from math import factorial

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.standard_tableaux.enumeration import (
    enumerate_standard_young_tableaux,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    require_standard,
)


def _hook_formula(parts: tuple[int, ...]) -> int:
    size = sum(parts)
    columns = tuple(
        sum(row_length > column for row_length in parts)
        for column in range(parts[0] if parts else 0)
    )
    hook_product = 1
    for row, row_length in enumerate(parts):
        for column in range(row_length):
            hook_product *= row_length - column + columns[column] - row - 1
    return factorial(size) // hook_product


def _partitions(n: int, maximum: int | None = None):
    if n == 0:
        yield ()
        return
    bound = min(n, n if maximum is None else maximum)
    for part in range(bound, 0, -1):
        for rest in _partitions(n - part, part):
            yield (part, *rest)


def test_enumeration_is_complete_for_every_shape_through_size_seven() -> None:
    for size in range(8):
        for parts in _partitions(size):
            shape = IntegerPartition(parts=parts)
            result = enumerate_standard_young_tableaux(shape)
            rows = tuple(tableau.rows for tableau in result.tableaux)

            assert result.partition == shape
            assert rows == tuple(sorted(rows))
            assert len(rows) == len(set(rows)) == _hook_formula(parts)
            assert all(tableau.shape == shape for tableau in result.tableaux)
            for tableau in result.tableaux:
                require_standard(tableau)


def test_known_shape_two_one_and_empty_shape() -> None:
    pair = enumerate_standard_young_tableaux(IntegerPartition(parts=(2, 1)))
    assert tuple(tableau.rows for tableau in pair.tableaux) == (
        ((1, 2), (3,)),
        ((1, 3), (2,)),
    )
    empty = enumerate_standard_young_tableaux(IntegerPartition(parts=()))
    assert tuple(tableau.rows for tableau in empty.tableaux) == ((),)


def test_complete_family_is_admitted_before_construction() -> None:
    with pytest.raises(
        OperationResourceAdmissionError,
        match="complete standard-tableau family exceeds the admitted count",
    ):
        enumerate_standard_young_tableaux(IntegerPartition(parts=(5, 5, 5)))
