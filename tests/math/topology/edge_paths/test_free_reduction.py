from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.topology.edge_paths._models import (
    FiniteGroupWord,
    FreeReductionRequest,
    WordLetter,
)
from jacobian.math.topology.edge_paths.operations import free_reduce

_OPERATION_ID = "topology.group_presentation.free_reduce.compute"
_ALPHABET = ((0, 1), (0, -1), (1, 1), (1, -1))


def _request(letters: tuple[tuple[int, int], ...]) -> FreeReductionRequest:
    return FreeReductionRequest(
        generator_count=2,
        letters=tuple(
            WordLetter(generator=generator, exponent=exponent)
            for generator, exponent in letters
        ),
    )


def _stack_oracle(
    letters: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Independent literal free-group normal-form oracle for bounded words."""
    reduced: list[tuple[int, int]] = []
    for letter in letters:
        if reduced and reduced[-1][0] == letter[0] and reduced[-1][1] == -letter[1]:
            reduced.pop()
        else:
            reduced.append(letter)
    return tuple(reduced)


def _pairs(word: FiniteGroupWord) -> tuple[tuple[int, int], ...]:
    return tuple((letter.generator, letter.exponent) for letter in word.letters)


def test_free_reduction_matches_exact_oracle_exhaustively() -> None:
    for length in range(8):
        for letters in itertools.product(_ALPHABET, repeat=length):
            result = free_reduce(_request(letters))
            assert _pairs(result) == _stack_oracle(letters)


def test_free_reduction_is_idempotent_and_commutes_with_word_inverse() -> None:
    for letters in (
        (),
        ((0, 1),),
        ((0, 1), (1, -1), (1, 1), (0, -1)),
        ((0, 1), (0, 1), (0, -1), (1, -1), (1, 1)),
    ):
        reduced = free_reduce(_request(letters))
        assert (
            free_reduce(
                FreeReductionRequest(generator_count=2, letters=reduced.letters)
            )
            == reduced
        )

        inverse_letters = tuple(
            (generator, -sign) for generator, sign in reversed(letters)
        )
        reduced_inverse = free_reduce(_request(inverse_letters))
        expected_inverse = tuple(
            (generator, -sign) for generator, sign in reversed(_pairs(reduced))
        )
        assert _pairs(reduced_inverse) == expected_inverse


def test_free_reduction_catalog_operation_and_serialized_result() -> None:
    tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == _OPERATION_ID)
    request = tool.request_type(
        generator_count=2,
        letters=[
            {"generator": 0, "exponent": 1},
            {"generator": 0, "exponent": -1},
            {"generator": 1, "exponent": 1},
        ],
    )
    result = tool.run(request)
    assert _pairs(result) == ((1, 1),)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_free_reduction_empty_and_maximum_words() -> None:
    assert (
        free_reduce(FreeReductionRequest(generator_count=0, letters=())).letters == ()
    )
    maximum = FreeReductionRequest(
        generator_count=1,
        letters=tuple(WordLetter(generator=0, exponent=1) for _ in range(128)),
    )
    assert len(free_reduce(maximum).letters) == 128
    with pytest.raises(ValidationError):
        FreeReductionRequest(
            generator_count=1,
            letters=tuple(WordLetter(generator=0, exponent=1) for _ in range(129)),
        )
    with pytest.raises(ValidationError):
        FreeReductionRequest(
            generator_count=0,
            letters=(WordLetter(generator=0, exponent=1),),
        )
