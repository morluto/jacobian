from __future__ import annotations

from itertools import combinations, pairwise, product

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.algebraic._models import (
    LongestIncreasingSubsequenceRequest,
)
from jacobian.math.combinatorics.algebraic._tools import TOOLS
from jacobian.math.combinatorics.algebraic.subsequences import (
    longest_increasing_subsequence,
)
from jacobian.math.logic.languages.words.values import FiniteWord


def _word(
    letters: tuple[str, ...], alphabet: tuple[str, ...] = ("a", "b", "c")
) -> FiniteWord:
    return FiniteWord(alphabet=alphabet, letters=letters)


def _oracle(word: FiniteWord) -> int:
    rank = {letter: index for index, letter in enumerate(word.alphabet)}
    best = 0
    for size in range(len(word.letters) + 1):
        for indices in combinations(range(len(word.letters)), size):
            values = tuple(rank[word.letters[index]] for index in indices)
            if all(left < right for left, right in pairwise(values)):
                best = max(best, size)
    return best


def test_exhaustive_short_words_match_independent_subsequence_oracle() -> None:
    alphabet = ("z", "a", "m")
    for length in range(6):
        for letters in product(alphabet, repeat=length):
            word = _word(letters, alphabet)
            result = longest_increasing_subsequence(
                LongestIncreasingSubsequenceRequest(word=word)
            )

            assert result.length == _oracle(word)
            assert len(result.indices) == result.length
            assert tuple(sorted(result.indices)) == result.indices
            assert result.values == tuple(word.letters[i] for i in result.indices)
            ranks = {letter: index for index, letter in enumerate(alphabet)}
            assert all(
                ranks[left] < ranks[right] for left, right in pairwise(result.values)
            )


@pytest.mark.parametrize(
    ("letters", "expected_indices", "expected_values"),
    [
        ((), (), ()),
        (("b",), (0,), ("b",)),
        (("b", "b", "b"), (0,), ("b",)),
        (("b", "a", "b", "c"), (1, 2, 3), ("a", "b", "c")),
    ],
)
def test_empty_singleton_duplicate_and_deterministic_tie_cases(
    letters: tuple[str, ...],
    expected_indices: tuple[int, ...],
    expected_values: tuple[str, ...],
) -> None:
    alphabet = ("a", "b", "c") if letters else ()
    result = longest_increasing_subsequence(
        LongestIncreasingSubsequenceRequest(word=_word(letters, alphabet))
    )
    assert result.indices == expected_indices
    assert result.values == expected_values
    assert result.length == len(expected_indices)


def test_tool_is_published_with_strict_convention_and_bound_example() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "word.longest_increasing_subsequence.compute"
    )
    assert "strictly greater" in tool.description
    assert "O(n^2)" in tool.description
    assert tool.examples[0].input["word"]["letters"] == ["b", "a", "b", "c"]


def test_oversized_forged_word_is_rejected_before_dynamic_programming() -> None:
    source = FiniteWord.model_construct(alphabet=("a",), letters=("a",) * 501)
    request = LongestIncreasingSubsequenceRequest.model_construct(word=source)
    with pytest.raises(OperationResourceAdmissionError, match="word length"):
        longest_increasing_subsequence(request)


def test_payload_cardinality_is_admitted_before_dynamic_programming() -> None:
    source = FiniteWord.model_construct(alphabet=("a" * 140_801,), letters=())
    request = LongestIncreasingSubsequenceRequest.model_construct(word=source)
    with pytest.raises(OperationResourceAdmissionError, match="payload exceeds"):
        longest_increasing_subsequence(request)
