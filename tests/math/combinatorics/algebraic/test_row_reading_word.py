from __future__ import annotations

import pytest

from jacobian.math.combinatorics.algebraic import operations
from jacobian.math.combinatorics.algebraic.values import RSKTableauPair
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
    StandardYoungTableau,
)
from jacobian.math.logic.languages.words.values import FiniteWord


def test_reads_existing_insertion_tableau_bottom_to_top_and_composes_with_rsk() -> None:
    pair = RSKTableauPair(
        alphabet=("a", "b", "c"),
        insertion_tableau=SemistandardYoungTableau(rows=((1, 2), (3,))),
        recording_tableau=StandardYoungTableau(rows=((1, 3), (2,))),
        shape=IntegerPartition(parts=(2, 1)),
    )

    read = operations.tableau_row_reading_word(pair)

    assert read == FiniteWord(alphabet=("a", "b", "c"), letters=("c", "a", "b"))
    # Independent row insertion establishes that reading the tableau recovers
    # a word whose P tableau is the supplied tableau.
    recovered_pair = operations.row_insertion_rsk(read)
    assert recovered_pair.insertion_tableau == pair.insertion_tableau


def test_empty_tableau_keeps_its_empty_alphabet() -> None:
    pair = RSKTableauPair(
        alphabet=(),
        insertion_tableau=SemistandardYoungTableau(rows=()),
        recording_tableau=StandardYoungTableau(rows=()),
        shape=IntegerPartition(parts=()),
    )

    assert operations.tableau_row_reading_word(pair) == FiniteWord(
        alphabet=(), letters=()
    )


def test_rejects_forged_pair_with_rank_outside_alphabet() -> None:
    pair = RSKTableauPair(
        alphabet=("a",),
        insertion_tableau=SemistandardYoungTableau(rows=((1,),)),
        recording_tableau=StandardYoungTableau(rows=((1,),)),
        shape=IntegerPartition(parts=(1,)),
    ).model_copy(
        update={
            "insertion_tableau": SemistandardYoungTableau(rows=((2,),)),
        }
    )

    with pytest.raises(ValueError):
        operations.tableau_row_reading_word(pair)
