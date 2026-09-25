from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.combinatorics.algebraic import (
    PlacticEquivalenceRequest,
    plactic_equivalence,
)
from jacobian.math.combinatorics.algebraic._tools import TOOLS
from jacobian.math.logic.languages.words.values import FiniteWord


def _word(letters: tuple[str, ...]) -> FiniteWord:
    return FiniteWord(alphabet=("a", "b", "c"), letters=letters)


def _independent_insertion(word: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    rank = {letter: index for index, letter in enumerate(("a", "b", "c"), start=1)}
    rows: list[list[int]] = []
    for symbol in word:
        letter = rank[symbol]
        bumped = letter
        for row in rows:
            column = next(
                (index for index, entry in enumerate(row) if entry > bumped),
                len(row),
            )
            if column == len(row):
                row.append(bumped)
                break
            row[column], bumped = bumped, row[column]
        else:
            rows.append([bumped])
    return tuple(tuple(row) for row in rows)


def test_knuth_generators_and_near_miss_have_exact_relation() -> None:
    first = plactic_equivalence(_word(("a", "c", "b")), _word(("c", "a", "b")))
    second = plactic_equivalence(_word(("b", "a", "c")), _word(("b", "c", "a")))
    near_miss = plactic_equivalence(_word(("a", "b", "c")), _word(("b", "a", "c")))

    assert first.equivalent and second.equivalent
    assert first.left.normal_form == first.right.normal_form
    assert second.left.insertion_tableau == second.right.insertion_tableau
    assert not near_miss.equivalent


def test_exhaustive_short_words_match_independent_insertion_oracle() -> None:
    words = [
        tuple(letters)
        for length in range(4)
        for letters in product(("a", "b", "c"), repeat=length)
    ]
    insertions = {word: _independent_insertion(word) for word in words}

    for left in words:
        for right in words:
            result = plactic_equivalence(_word(left), _word(right))
            expected = insertions[left] == insertions[right]
            assert result.equivalent is expected
            assert result.left.insertion_tableau.rows == insertions[left]
            assert result.right.insertion_tableau.rows == insertions[right]


def test_plactic_equivalence_requires_the_same_ordered_alphabet() -> None:
    left = FiniteWord(alphabet=("a", "b"), letters=("a",))
    right = FiniteWord(alphabet=("b", "a"), letters=("a",))
    with pytest.raises(ValidationError):
        PlacticEquivalenceRequest(left=left, right=right)


def test_native_operation_accepts_canonical_words_and_rejects_forged_sources() -> None:
    left = FiniteWord.model_construct(alphabet=("a", "b"), letters=("a",))
    right = FiniteWord.model_construct(alphabet=("b", "a"), letters=("a",))
    with pytest.raises(OperationDomainValidationError) as error:
        plactic_equivalence(left, right)
    assert error.value.errors()[0]["type"] == (
        "algebraic_combinatorics.plactic_equivalence_request"
    )


def test_equivalence_contract_round_trips_and_is_catalogued() -> None:
    request = PlacticEquivalenceRequest(
        left=_word(("a", "c", "b")), right=_word(("c", "a", "b"))
    )
    result = plactic_equivalence(request.left, request.right)
    reconstructed = type(result).model_validate_json(result.model_dump_json())

    assert reconstructed == result
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "word.plactic_equivalence.compute"
    )
    assert tool.examples
    example_request = PlacticEquivalenceRequest.model_validate(tool.examples[0].input)
    example_result = tool.run(example_request)
    assert example_result.equivalent is True
    catalog = Catalog.open()
    operation = catalog.operation(tool.operation_id)
    assert operation is not None
    invoked = invoke_operation(tool.operation_id, tool.examples[0].input, catalog)
    assert operation.result_type.model_validate(invoked.output).equivalent is True
