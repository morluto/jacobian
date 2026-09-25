from itertools import product

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.semistandard_tableaux._models import (
    SemistandardTableauEnumerationRequest,
)
from jacobian.math.combinatorics.semistandard_tableaux.enumeration import (
    enumerate_semistandard_young_tableaux,
    semistandard_tableaux_count,
)
from jacobian.math.combinatorics.symmetric_functions.values import IntegerPartition


def _brute_force(
    parts: tuple[int, ...], max_entry: int
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    cells = tuple(
        (row, column) for row, width in enumerate(parts) for column in range(width)
    )
    tableaux = []
    for entries in product(range(1, max_entry + 1), repeat=len(cells)):
        rows = [[0] * width for width in parts]
        for (row, column), entry in zip(cells, entries, strict=True):
            rows[row][column] = entry
        if any(
            row[column] > row[column + 1]
            for row in rows
            for column in range(len(row) - 1)
        ):
            continue
        if any(
            rows[row][column] >= rows[row + 1][column]
            for row in range(len(rows) - 1)
            for column in range(len(rows[row + 1]))
        ):
            continue
        tableaux.append(tuple(tuple(row) for row in rows))
    return tuple(tableaux)


@pytest.mark.parametrize(
    ("parts", "max_entry"),
    [
        ((), 0),
        ((1,), 0),
        ((1, 1), 1),
        ((2, 1), 2),
        ((2, 1), 3),
        ((3, 2), 3),
    ],
)
def test_enumeration_matches_brute_force_and_hook_content_count(
    parts: tuple[int, ...], max_entry: int
) -> None:
    partition = IntegerPartition(parts=parts)

    result = enumerate_semistandard_young_tableaux(partition, max_entry)

    expected = _brute_force(parts, max_entry)
    assert tuple(tableau.rows for tableau in result.tableaux) == expected
    assert len(result.tableaux) == semistandard_tableaux_count(partition, max_entry)


def test_exact_maximum_single_cell_family_is_accepted() -> None:
    partition = IntegerPartition(parts=(1,))

    result = enumerate_semistandard_young_tableaux(partition, max_entry=4_096)

    assert len(result.tableaux) == 4_096
    assert result.tableaux[0].rows == ((1,),)
    assert result.tableaux[-1].rows == ((4_096,),)


def test_count_over_admitted_family_refuses_before_tableau_construction() -> None:
    with pytest.raises(OperationResourceAdmissionError) as error:
        enumerate_semistandard_young_tableaux(
            IntegerPartition(parts=(2,)), max_entry=91
        )

    assert error.value.errors()[0]["type"] == "semistandard_tableaux.count_bound"


def test_exact_maximum_cell_shape_is_accepted_with_one_tableau() -> None:
    result = enumerate_semistandard_young_tableaux(
        IntegerPartition(parts=(500,)), max_entry=1
    )

    assert result.tableaux[0].rows == ((1,) * 500,)


def test_catalog_request_keeps_alphabet_bound() -> None:
    request = SemistandardTableauEnumerationRequest(
        partition=IntegerPartition(parts=(1,)), max_entry=4_096
    )

    assert request.max_entry == 4_096
