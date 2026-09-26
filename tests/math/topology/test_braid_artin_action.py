"""Exact regression fixtures for the bounded Artin braid action."""

from __future__ import annotations

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.edge_paths._models import FiniteGroupWord, WordLetter
from jacobian.math.topology.links import BraidLetter, BraidWord, braid_artin_action
from jacobian.math.topology.links._extensions_models import BraidWordRequest


def _word(strands: int, *letters: tuple[int, int]) -> BraidWord:
    return BraidWord(
        strand_count=strands,
        letters=tuple(
            BraidLetter(generator=generator, exponent=exponent)
            for generator, exponent in letters
        ),
    )


def _free_word(*letters: tuple[int, int]) -> FiniteGroupWord:
    return FiniteGroupWord(
        letters=tuple(
            WordLetter(generator=generator, exponent=exponent)
            for generator, exponent in letters
        )
    )


def test_positive_and_negative_artin_generators_are_mutual_inverses() -> None:
    positive = braid_artin_action(_word(2, (1, 1)))
    negative = braid_artin_action(_word(2, (1, -1)))
    assert positive.generator_images == (
        _free_word((0, 1), (1, 1), (0, -1)),
        _free_word((0, 1)),
    )
    assert negative.generator_images == (
        _free_word((1, 1)),
        _free_word((1, -1), (0, 1), (1, 1)),
    )

    identity = braid_artin_action(_word(2, (1, 1), (1, -1)))
    assert identity.generator_images == (_free_word((0, 1)), _free_word((1, 1)))


def test_artin_action_obeys_braid_relation_and_trefoil_closure_permutation() -> None:
    left = braid_artin_action(_word(3, (1, 1), (2, 1), (1, 1)))
    right = braid_artin_action(_word(3, (2, 1), (1, 1), (2, 1)))
    assert left.generator_images == right.generator_images

    trefoil_braid = _word(2, (1, 1), (1, 1), (1, 1))
    action = braid_artin_action(trefoil_braid)
    exponent_vectors = tuple(
        tuple(
            sum(letter.exponent for letter in image.letters if letter.generator == j)
            for j in range(2)
        )
        for image in action.generator_images
    )
    # Abelianization is the strand permutation; sigma_1^3 swaps the meridians.
    assert exponent_vectors == ((0, 1), (1, 0))


def test_action_admits_cancellation_and_bounds_reduced_expansion() -> None:
    cancelled = braid_artin_action(_word(2, *((1, 1),) * 6))
    assert tuple(len(image.letters) for image in cancelled.generator_images) == (13, 11)
    with pytest.raises(OperationResourceAdmissionError, match="128-letter"):
        braid_artin_action(_word(2, *((1, 1),) * 64))


def test_action_is_published_as_one_typed_catalog_operation() -> None:
    tools = {tool.operation_id: tool for tool in BUILTIN_TOOLS}
    operation = tools["braid.word.artin_action.compute"]
    result = operation.run(BraidWordRequest(word=_word(2, (1, 1), (1, -1))))
    assert result.generator_images == (_free_word((0, 1)), _free_word((1, 1)))


def test_action_reduces_inverse_braid_prefixes_before_bounded_expansion() -> None:
    # w=(sigma_2^2 sigma_1^3)^2 sigma_2^2 followed by w^-1.
    w = ((2, 1), (2, 1), (1, 1), (1, 1), (1, 1)) * 2 + ((2, 1), (2, 1))
    inverse = tuple((generator, -exponent) for generator, exponent in reversed(w))
    letters = tuple(
        (generator, 1 if exponent == 1 else -1) for generator, exponent in w + inverse
    )
    result = braid_artin_action(_word(3, *letters))
    assert result.generator_images == tuple(_free_word((i, 1)) for i in range(3))
