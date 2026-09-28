"""Contract tests for exact free associative words and NC polynomials."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.free_algebras._models import (
    MAX_FREE_ALGEBRA_GENERATORS,
    MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH,
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
    FreeAlgebraWord,
    canonical_word_key,
)


def _term(coefficient: int | Fraction, word: tuple[str, ...]) -> FreeAlgebraTerm:
    return FreeAlgebraTerm(
        coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
        word=word,
    )


def _poly(
    alphabet: tuple[str, ...], coefficients: dict[tuple[str, ...], int | Fraction]
) -> FreeAlgebraPolynomial:
    ordered = sorted(
        coefficients.items(),
        key=lambda item: canonical_word_key(alphabet, item[0]),
        reverse=True,
    )
    return FreeAlgebraPolynomial(
        alphabet=alphabet,
        terms=tuple(_term(value, word) for word, value in ordered if value != 0),
    )


def test_word_value_is_bounded_and_bound_to_its_alphabet() -> None:
    word = FreeAlgebraWord(alphabet=("x", "y"), letters=("y", "x", "x"))
    assert word.length == 3
    assert not word.is_unit

    unit = FreeAlgebraWord(alphabet=("x", "y"), letters=())
    assert unit.length == 0
    assert unit.is_unit

    empty_alphabet_unit = FreeAlgebraWord(alphabet=(), letters=())
    assert empty_alphabet_unit.is_unit


def test_word_value_rejects_structural_boundaries() -> None:
    with pytest.raises(ValidationError):
        FreeAlgebraWord(alphabet=("x", "x"), letters=("x",))
    with pytest.raises(ValidationError):
        FreeAlgebraWord(alphabet=("x",), letters=("z",))
    value_bound = FreeAlgebraWord(
        alphabet=tuple("abcdefghijklmnopqrstuvwxyz"),
        letters=("a",) * MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH,
    )
    assert value_bound.length == MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH
    with pytest.raises(ValidationError):
        FreeAlgebraWord(
            alphabet=tuple("abcdefghijklmnopqrstuvwxyz"),
            letters=("a",) * (MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH + 1),
        )
    with pytest.raises(ValidationError):
        FreeAlgebraWord(
            alphabet=tuple(
                f"g{index}" for index in range(MAX_FREE_ALGEBRA_GENERATORS + 1)
            )
        )


def test_word_canonical_key_is_degree_then_generator_rank() -> None:
    alphabet = ("x", "y", "z")
    assert canonical_word_key(alphabet, ("z",)) > canonical_word_key(alphabet, ("x",))
    assert canonical_word_key(alphabet, ("x", "x")) > canonical_word_key(
        alphabet, ("z",)
    )
    assert canonical_word_key(alphabet, ("x", "y")) < canonical_word_key(
        alphabet, ("y", "x")
    )


def test_polynomial_collects_like_words_and_orders_canonically() -> None:
    polynomial = _poly(
        ("x", "y"),
        {
            ("x",): Fraction(1),
            ("y",): Fraction(1),
            ("x", "x"): Fraction(-2),
            ("y", "x"): Fraction(1, 2),
        },
    )
    assert tuple(term.word for term in polynomial.terms) == (
        ("y", "x"),
        ("x", "x"),
        ("y",),
        ("x",),
    )
    assert polynomial.terms[0].coefficient.as_fraction() == Fraction(1, 2)


def test_polynomial_rejects_uncanonical_structure() -> None:
    # Duplicate words are not collected.
    with pytest.raises(ValidationError):
        FreeAlgebraPolynomial(
            alphabet=("x",),
            terms=(_term(1, ("x",)), _term(2, ("x",))),
        )
    # Zero terms are omitted.
    with pytest.raises(ValidationError):
        FreeAlgebraPolynomial(alphabet=("x",), terms=(_term(0, ("x",)),))
    # Ascending order is rejected.
    with pytest.raises(ValidationError):
        FreeAlgebraPolynomial(
            alphabet=("x", "y"),
            terms=(_term(1, ("x",)), _term(1, ("y",))),
        )
    # A word outside the declared alphabet is rejected.
    with pytest.raises(ValidationError):
        FreeAlgebraPolynomial(alphabet=("x",), terms=(_term(1, ("y",)),))
    maximal = FreeAlgebraPolynomial(
        alphabet=("x",),
        terms=(_term(1, ("x",) * MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH),),
    )
    assert len(maximal.terms[0].word) == MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH
    with pytest.raises(ValidationError):
        FreeAlgebraPolynomial(
            alphabet=("x",),
            terms=(_term(1, ("x",) * (MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH + 1)),),
        )


def test_zero_polynomial_is_closed_and_retains_its_alphabet() -> None:
    zero = _poly(("x", "y"), {})
    assert zero.is_zero
    assert zero.alphabet == ("x", "y")
    replayed = FreeAlgebraPolynomial.model_validate_json(zero.model_dump_json())
    assert replayed == zero


def test_word_and_polynomial_round_trip_through_json() -> None:
    word = FreeAlgebraWord(alphabet=("x", "y"), letters=("x", "y", "x"))
    assert FreeAlgebraWord.model_validate_json(word.model_dump_json()) == word

    polynomial = _poly(
        ("x", "y"),
        {("x",): Fraction(3, 2), ("y", "x"): Fraction(-1, 4)},
    )
    assert (
        FreeAlgebraPolynomial.model_validate_json(polynomial.model_dump_json())
        == polynomial
    )
