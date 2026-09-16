"""Tests for one-step Knuth moves (#1769)."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.algebraic._models import (
    KnuthMovesRequest,
    KnuthMovesResult,
)
from jacobian.math.combinatorics.algebraic._tools import knuth_moves as catalog_knuth
from jacobian.math.combinatorics.algebraic.operations import (
    knuth_moves,
    row_insertion_rsk,
)
from jacobian.math.logic.languages.words.values import FiniteWord


def _word(letters: tuple[str, ...]) -> FiniteWord:
    return FiniteWord(alphabet=("a", "b", "c"), letters=letters)


def _request(letters: tuple[str, ...]) -> KnuthMovesRequest:
    return KnuthMovesRequest(word=_word(letters))


class TestKnownAnswer:
    def test_k1_forward(self) -> None:
        result = catalog_knuth(_request(("a", "c", "b")))
        assert isinstance(result, KnuthMovesResult)
        assert result.neighbor_count == 1
        (neighbor,) = result.neighbors
        assert neighbor.position == 0
        assert neighbor.relation == "K1"
        assert neighbor.neighbor.letters == ("c", "a", "b")

    def test_k1_backward(self) -> None:
        (neighbor,) = knuth_moves(_word(("c", "a", "b")))
        assert neighbor.relation == "K1"
        assert neighbor.neighbor.letters == ("a", "c", "b")

    def test_k2_forward(self) -> None:
        (neighbor,) = knuth_moves(_word(("b", "a", "c")))
        assert neighbor.relation == "K2"
        assert neighbor.neighbor.letters == ("b", "c", "a")

    def test_k2_backward(self) -> None:
        (neighbor,) = knuth_moves(_word(("b", "c", "a")))
        assert neighbor.relation == "K2"
        assert neighbor.neighbor.letters == ("b", "a", "c")


class TestBoundaryDegenerate:
    @pytest.mark.parametrize("letters", [(), ("a",), ("a", "b")])
    def test_short_words_have_no_neighbors(self, letters: tuple[str, ...]) -> None:
        assert knuth_moves(_word(letters)) == ()

    def test_constant_word_has_no_neighbors(self) -> None:
        assert knuth_moves(_word(("a", "a", "a"))) == ()

    def test_strictly_increasing_word_has_no_neighbors(self) -> None:
        assert knuth_moves(_word(("a", "b", "c"))) == ()


class TestAdversarial:
    def test_letter_outside_alphabet_rejected(self) -> None:
        forged = FiniteWord.model_construct(
            alphabet=("a", "b", "c"), letters=("a", "z", "b")
        )
        with pytest.raises(OperationDomainValidationError):
            knuth_moves(forged)

    def test_duplicate_alphabet_rejected(self) -> None:
        forged = FiniteWord.model_construct(
            alphabet=("a", "a", "b"), letters=("a", "a", "b")
        )
        with pytest.raises(OperationDomainValidationError):
            knuth_moves(forged)


class TestDefiningInvariant:
    @pytest.mark.parametrize(
        "letters",
        [
            ("a", "c", "b"),
            ("c", "a", "b"),
            ("b", "a", "c"),
            ("b", "c", "a"),
            ("a", "c", "b", "a"),
            ("c", "b", "a", "c"),
        ],
    )
    def test_neighbors_differ_by_one_valid_relation(
        self, letters: tuple[str, ...]
    ) -> None:
        word = _word(letters)
        rank = {symbol: index for index, symbol in enumerate(word.alphabet)}
        for entry in knuth_moves(word):
            i = entry.position
            before = tuple(rank[s] for s in word.letters[i : i + 3])
            after_letters = entry.neighbor.letters
            assert after_letters[:i] == word.letters[:i]
            assert after_letters[i + 3 :] == word.letters[i + 3 :]
            a, b, c = before
            if entry.relation == "K1":
                assert a <= c < b or b <= c < a
                assert after_letters[i : i + 3] == (
                    word.letters[i + 1],
                    word.letters[i],
                    word.letters[i + 2],
                )
            else:
                assert b < a <= c or c < a <= b
                assert after_letters[i : i + 3] == (
                    word.letters[i],
                    word.letters[i + 2],
                    word.letters[i + 1],
                )

    @pytest.mark.parametrize(
        "letters",
        [
            ("a", "c", "b"),
            ("c", "a", "b"),
            ("b", "a", "c"),
            ("b", "c", "a"),
            ("a", "c", "b", "a"),
        ],
    )
    def test_neighbors_share_insertion_tableau(self, letters: tuple[str, ...]) -> None:
        word = _word(letters)
        source_tableau = row_insertion_rsk(word).insertion_tableau
        for entry in knuth_moves(word):
            assert row_insertion_rsk(entry.neighbor).insertion_tableau == source_tableau


class TestNativeCatalogParity:
    @pytest.mark.parametrize(
        "letters", [(("a", "c", "b")), (("b", "a", "c")), (("a", "b", "c"))]
    )
    def test_native_matches_catalog(self, letters: tuple[str, ...]) -> None:
        request = _request(letters)
        assert catalog_knuth(request).neighbors == knuth_moves(request.word)
