"""Correctness and contract tests for bounded combinatorics on words."""

from __future__ import annotations

import itertools
import random

import pytest
from pydantic import ValidationError

from jacobian.math.words import (
    FiniteWord,
    WordMorphism,
    apply_morphism,
    compose_morphisms,
    conjugates,
    factor_occurrences,
    factors_of_length,
    incidence_matrix,
    parikh_vector,
    periods,
    prefix_function,
    primitive_root,
)
from jacobian.math.words._models import (
    FactorsLengthRequest,
    FactorsLengthResult,
    IncidenceMatrixRequest,
    IncidenceMatrixResult,
    PeriodsRequest,
    PeriodsResult,
)
from jacobian.math.words._operations import (
    compute_factors_length,
    compute_incidence_matrix,
    compute_periods,
)
from jacobian.math.words._tools import TOOLS


def _word(letters: str, alphabet: tuple[str, ...] = ("a", "b")) -> FiniteWord:
    return FiniteWord(alphabet=alphabet, letters=tuple(letters))


def test_public_catalog_surface_is_the_audited_three_operations() -> None:
    assert tuple(tool.operation_id for tool in TOOLS) == (
        "word.factors.length.compute",
        "word.periods.compute",
        "word_morphism.incidence_matrix.compute",
    )


def test_factor_result_is_complete_and_bound_to_the_request() -> None:
    request = FactorsLengthRequest(word=_word("abaab"), factor_length=2)
    result = compute_factors_length(request)
    assert result.factors == (("a", "b"), ("b", "a"), ("a", "a"))
    assert result.occurrences == ((0, 3), (1,), (2,))
    assert result.multiplicities == (2, 1, 1)
    assert result.first_occurrence == (0, 1, 2)
    assert result.distinct_count == 3
    assert result.complete is True

    payload = result.model_dump()
    payload["distinct_count"] = 2
    with pytest.raises(ValidationError, match="not bound"):
        FactorsLengthResult.model_validate(payload)


def test_empty_factor_occurs_at_every_boundary() -> None:
    result = compute_factors_length(
        FactorsLengthRequest(word=_word("aa", ("a",)), factor_length=0)
    )
    assert result.factors == ((),)
    assert result.occurrences == ((0, 1, 2),)


def test_factor_length_is_validated_before_computation() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        FactorsLengthRequest(word=_word("aa", ("a",)), factor_length=3)


def test_periods_distinguish_overlap_period_from_proper_power() -> None:
    repeated = compute_periods(PeriodsRequest(word=_word("ababab")))
    assert repeated.periods == (2, 4, 6)
    assert repeated.least_period == 2
    assert repeated.is_primitive is False

    bordered_but_primitive = compute_periods(PeriodsRequest(word=_word("ababa")))
    assert bordered_but_primitive.periods == (2, 4, 5)
    assert bordered_but_primitive.least_period == 2
    assert bordered_but_primitive.is_primitive is True


def test_empty_period_convention_and_result_binding() -> None:
    result = compute_periods(PeriodsRequest(word=_word("", ("a",))))
    assert result.periods == ()
    assert result.least_period == 0
    assert result.is_primitive is False

    payload = result.model_dump()
    payload["is_primitive"] = True
    with pytest.raises(ValidationError, match="not bound"):
        PeriodsResult.model_validate(payload)


def test_fibonacci_incidence_matrix_and_binding() -> None:
    morphism = WordMorphism(
        source_alphabet=("a", "b"),
        target_alphabet=("a", "b"),
        images=(("a", "b"), ("a",)),
    )
    result = compute_incidence_matrix(IncidenceMatrixRequest(morphism=morphism))
    assert result.matrix == ((1, 1), (1, 0))
    assert result.orientation == "ROWS_TARGET_COLUMNS_SOURCE"

    payload = result.model_dump()
    payload["matrix"] = ((1, 0), (1, 1))
    with pytest.raises(ValidationError, match="not bound"):
        IncidenceMatrixResult.model_validate(payload)


def test_native_word_operations_are_exact_and_use_declared_order() -> None:
    word = _word("aaa", ("a",))
    assert factor_occurrences(word, ("a", "a")) == (0, 1)
    assert factor_occurrences(word, ()) == (0, 1, 2, 3)
    assert primitive_root(_word("abcabc", ("a", "b", "c"))) == (
        ("a", "b", "c"),
        2,
    )
    assert conjugates(_word("abb", ("b", "a"))) == (
        ("b", "b", "a"),
        ("b", "a", "b"),
        ("a", "b", "b"),
    )
    assert parikh_vector(_word("abaab")) == (3, 2)
    assert prefix_function(_word("aabaab")) == (0, 1, 0, 1, 2, 3)


def test_native_morphism_application_and_composition() -> None:
    fibonacci = WordMorphism(
        source_alphabet=("a", "b"),
        target_alphabet=("a", "b"),
        images=(("a", "b"), ("a",)),
    )
    swap = WordMorphism(
        source_alphabet=("a", "b"),
        target_alphabet=("x", "y"),
        images=(("y",), ("x",)),
    )
    assert apply_morphism(fibonacci, _word("ab")).letters == ("a", "b", "a")
    composed = compose_morphisms(fibonacci, swap)
    assert composed.images == (("y", "x"), ("y",))
    assert incidence_matrix(composed) == ((1, 0), (1, 1))


def test_value_models_reject_ambiguous_or_unbounded_inputs() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        FiniteWord(alphabet=("a", "a"), letters=("a",))
    with pytest.raises(ValidationError, match="outside"):
        FiniteWord(alphabet=("a",), letters=("b",))

    expanding = WordMorphism(
        source_alphabet=("a",),
        target_alphabet=("a",),
        images=(("a",) * 2,),
    )
    with pytest.raises(ValueError, match="output exceeds"):
        apply_morphism(
            expanding,
            FiniteWord(alphabet=("a",), letters=("a",) * 500),
        )


def test_random_words_match_independent_factor_and_period_oracles() -> None:
    random_source = random.Random(1966)
    for length in range(9):
        for _ in range(40):
            letters = tuple(random_source.choice(("a", "b")) for _ in range(length))
            word = FiniteWord(alphabet=("a", "b"), letters=letters)
            for factor_length in range(length + 1):
                analysis = factors_of_length(word, factor_length)
                windows = tuple(
                    letters[index : index + factor_length]
                    for index in range(length - factor_length + 1)
                )
                expected_factors = tuple(dict.fromkeys(windows))
                expected_positions = tuple(
                    tuple(
                        index
                        for index, window in enumerate(windows)
                        if window == factor
                    )
                    for factor in expected_factors
                )
                assert analysis.factors == expected_factors
                assert analysis.occurrences == expected_positions

            analysis = periods(word)
            expected_periods = tuple(
                period
                for period in range(1, length + 1)
                if letters[:-period] == letters[period:]
            )
            proper_power = any(
                length % root_length == 0
                and letters[:root_length] * (length // root_length) == letters
                for root_length in range(1, length)
            )
            assert analysis.periods == expected_periods
            assert analysis.primitive is (length > 0 and not proper_power)


def test_incidence_matrix_matches_independent_count_oracle() -> None:
    alphabet = ("a", "b")
    images = tuple(itertools.product(alphabet, repeat=2))
    for left in images:
        for right in images:
            morphism = WordMorphism(
                source_alphabet=alphabet,
                target_alphabet=alphabet,
                images=(left, right),
            )
            assert incidence_matrix(morphism) == tuple(
                tuple(image.count(target) for image in (left, right))
                for target in alphabet
            )
