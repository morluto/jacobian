from __future__ import annotations

import pytest

from jacobian.math.combinatorics.algebraic import operations
from jacobian.math.combinatorics.algebraic._tools import TOOLS
from jacobian.math.combinatorics.algebraic.values import (
    MAX_RSK_WORD_LENGTH,
    RSKTableauPair,
)
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


def _wide_ascii_alphabet(count: int) -> tuple[str, ...]:
    return tuple(f"s{index:02d}".ljust(64, "x") for index in range(count))


def test_large_alphabet_with_small_tableau_is_admitted_and_read_exactly() -> None:
    # A 50-symbol alphabet of 64-character ASCII strings with an 11-cell
    # tableau emits only the alphabet once plus eleven referenced symbols;
    # admission must estimate the emitted payload, not every cell against the
    # whole alphabet.
    alphabet = _wide_ascii_alphabet(50)
    pair = RSKTableauPair(
        alphabet=alphabet,
        insertion_tableau=SemistandardYoungTableau(
            rows=(tuple(range(1, 12)),),
        ),
        recording_tableau=StandardYoungTableau(rows=(tuple(range(1, 12)),)),
        shape=IntegerPartition(parts=(11,)),
    )

    read = operations.tableau_row_reading_word(pair)

    assert read.letters == alphabet[:11]
    assert read.alphabet == alphabet
    # Shape-preserving invariant: reinserting the row reading recovers P.
    recovered = operations.row_insertion_rsk(read)
    assert recovered.insertion_tableau == pair.insertion_tableau
    assert recovered.shape == pair.shape
    assert operations.inverse_row_insertion_rsk(pair) == read


def test_row_reading_admits_the_full_cell_envelope_over_a_wide_alphabet() -> None:
    alphabet = _wide_ascii_alphabet(50)
    word = FiniteWord(alphabet=alphabet, letters=(alphabet[0],) * MAX_RSK_WORD_LENGTH)
    pair = operations.row_insertion_rsk(word)
    assert pair.shape.parts == (MAX_RSK_WORD_LENGTH,)

    read = operations.tableau_row_reading_word(pair)

    assert read == word
    assert operations.row_insertion_rsk(read) == pair


def test_row_reading_projection_stays_a_native_helper() -> None:
    # Publishing the cheap deterministic row-reading projection as a catalog
    # operation would add a redundant discovery intent over RSKTableauPair.
    assert all(
        tool.operation_id != "tableau.row_reading_word.compute" for tool in TOOLS
    )
