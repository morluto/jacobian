"""Structural contracts for the canonical Young-tableau values."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math import symmetric_functions
from jacobian.math.symmetric_functions import (
    IntegerPartition,
    SemistandardYoungTableau,
    StandardYoungTableau,
)
from jacobian.math.symmetric_functions._models import (
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
    ("rows", "message"),
    [
        (((1,), (2, 3)), "weakly decreasing"),
        (((1, 3, 2),), "weakly increasing"),
        (((1, 2), (1,)), "columns must be strictly increasing"),
    ],
)
def test_semistandard_tableau_rejects_malformed_rows(
    rows: tuple[tuple[int, ...], ...], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        SemistandardYoungTableau(rows=rows)


def test_standard_tableau_requires_exact_labels_and_strict_rows() -> None:
    tableau = StandardYoungTableau(rows=((1, 2, 4), (3,), (5,)))
    assert tableau.shape.parts == (3, 1, 1)

    with pytest.raises(ValidationError, match="exactly 1 through n"):
        StandardYoungTableau(rows=((1, 3),))
    with pytest.raises(ValidationError, match="strictly increasing"):
        StandardYoungTableau(rows=((1, 1),))


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
