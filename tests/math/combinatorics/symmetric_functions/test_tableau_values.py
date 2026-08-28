"""Structural contracts for the canonical Young-tableau values."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics import symmetric_functions
from jacobian.math.combinatorics.symmetric_functions import (
    IntegerPartition,
    SemistandardYoungTableau,
    StandardYoungTableau,
)
from jacobian.math.combinatorics.symmetric_functions._models import (
    IntegerPartition as RequestIntegerPartition,
)


def test_symmetric_function_public_values_have_one_canonical_identity() -> None:
    assert RequestIntegerPartition is IntegerPartition
    assert tuple(symmetric_functions.__all__) == (
        "IntegerPartition",
        "SemistandardYoungTableau",
        "StandardYoungTableau",
    )


def test_empty_tableaux_have_the_empty_partition_shape() -> None:
    assert SemistandardYoungTableau(rows=()).shape == IntegerPartition(parts=())
    assert StandardYoungTableau(rows=()).shape == IntegerPartition(parts=())


def test_semistandard_tableau_accepts_weak_rows_and_strict_columns() -> None:
    tableau = SemistandardYoungTableau(rows=((1, 1, 3), (2, 3), (4,)))
    assert tableau.shape.parts == (3, 2, 1)


@pytest.mark.parametrize(
    ("rows", "error_type"),
    [
        (((1,), (2, 3)), "symmetric_function.partition_not_weakly_decreasing"),
        (((1, 3, 2),), "symmetric_function.semistandard_rows_not_weakly_increasing"),
        (((1, 2), (1,)), "symmetric_function.tableau_columns_not_strict"),
    ],
)
def test_semistandard_tableau_rejects_malformed_rows(
    rows: tuple[tuple[int, ...], ...], error_type: str
) -> None:
    with pytest.raises(ValidationError) as error:
        SemistandardYoungTableau(rows=rows)
    assert error.value.errors()[0]["type"] == error_type


def test_standard_tableau_requires_exact_labels_and_strict_rows() -> None:
    tableau = StandardYoungTableau(rows=((1, 2, 4), (3,), (5,)))
    assert tableau.shape.parts == (3, 1, 1)

    with pytest.raises(ValidationError) as error:
        StandardYoungTableau(rows=((1, 3),))
    assert (
        error.value.errors()[0]["type"]
        == "symmetric_function.standard_entries_not_consecutive"
    )
    with pytest.raises(ValidationError) as error:
        StandardYoungTableau(rows=((1, 1),))
    assert (
        error.value.errors()[0]["type"]
        == "symmetric_function.standard_rows_not_strictly_increasing"
    )


def test_tableau_entries_are_strict_positive_integers() -> None:
    with pytest.raises(ValidationError):
        IntegerPartition(parts=(True,))
    with pytest.raises(ValidationError):
        SemistandardYoungTableau(rows=((True,),))
    with pytest.raises(ValidationError):
        SemistandardYoungTableau(rows=((0,),))


def test_semistandard_labels_are_not_bounded_by_tableau_size() -> None:
    assert SemistandardYoungTableau(rows=((101,),)).rows == ((101,),)
    assert SemistandardYoungTableau(rows=((2**53 - 1,),)).rows == ((2**53 - 1,),)
    with pytest.raises(ValidationError):
        SemistandardYoungTableau(rows=((2**53,),))


def test_tableau_shape_domain_is_closed_under_transposition_extremes() -> None:
    assert (
        SemistandardYoungTableau(
            rows=tuple((row,) for row in range(1, 101))
        ).shape.parts
        == (1,) * 100
    )
