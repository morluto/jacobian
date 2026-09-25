"""Exact free-algebra commutator tests with a direct word-convolution oracle."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.free_algebras._models import (
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
)
from jacobian.math.free_algebras.commutator._models import (
    FreeAlgebraCommutatorRequest,
    FreeAlgebraCommutatorResult,
)
from jacobian.math.free_algebras.commutator._tools import TOOLS
from jacobian.math.free_algebras.commutator.operations import commutator


def _polynomial(
    terms: tuple[tuple[tuple[str, ...], Fraction], ...],
) -> FreeAlgebraPolynomial:
    ordered = sorted(terms, key=lambda item: (len(item[0]), item[0]), reverse=True)
    return FreeAlgebraPolynomial(
        alphabet=("x", "y"),
        terms=tuple(
            FreeAlgebraTerm(
                word=word,
                coefficient=CanonicalRational.from_fraction(coefficient),
            )
            for word, coefficient in ordered
        ),
    )


def _word_product(
    left: FreeAlgebraPolynomial, right: FreeAlgebraPolynomial
) -> dict[tuple[str, ...], Fraction]:
    result: defaultdict[tuple[str, ...], Fraction] = defaultdict(Fraction)
    for left_term in left.terms:
        for right_term in right.terms:
            result[left_term.word + right_term.word] += (
                left_term.coefficient.as_fraction()
                * right_term.coefficient.as_fraction()
            )
    return {word: value for word, value in result.items() if value}


def _oracle_commutator(
    left: FreeAlgebraPolynomial, right: FreeAlgebraPolynomial
) -> dict[tuple[str, ...], Fraction]:
    forward = _word_product(left, right)
    backward = _word_product(right, left)
    words = set(forward) | set(backward)
    return {
        word: forward.get(word, Fraction()) - backward.get(word, Fraction())
        for word in words
        if forward.get(word, Fraction()) != backward.get(word, Fraction())
    }


def _coefficient_map(value: FreeAlgebraPolynomial) -> dict[tuple[str, ...], Fraction]:
    return {term.word: term.coefficient.as_fraction() for term in value.terms}


def _dense_operand() -> FreeAlgebraPolynomial:
    alphabet = tuple("abcdefghijklmnopqrstuvwxyz")
    words = tuple((alphabet[index // 26], alphabet[index % 26]) for index in range(64))
    ranks = {letter: rank for rank, letter in enumerate(alphabet)}
    return FreeAlgebraPolynomial(
        alphabet=alphabet,
        terms=tuple(
            FreeAlgebraTerm(
                word=word,
                coefficient=CanonicalRational.from_fraction(Fraction(1)),
            )
            for word in sorted(
                words,
                key=lambda word: (len(word), tuple(ranks[letter] for letter in word)),
                reverse=True,
            )
        ),
    )


def test_commutator_matches_independent_word_product_difference() -> None:
    left = _polynomial(
        (
            (("x",), Fraction(1, 2)),
            (("y",), Fraction(2, 3)),
            (("x", "y"), Fraction(-1, 4)),
        )
    )
    right = _polynomial(
        (
            (("x",), Fraction(3, 5)),
            (("y",), Fraction(-2, 7)),
            (("y", "x"), Fraction(1, 3)),
        )
    )

    result = commutator(left, right)

    actual = _coefficient_map(result.commutator)
    assert actual == _oracle_commutator(left, right)
    assert actual.get(("x", "x"), Fraction()) == 0
    assert result.commutator.alphabet == left.alphabet


def test_commutator_zero_identity_and_serialization() -> None:
    value = _polynomial(((("x",), Fraction(2, 3)), (("y", "x"), Fraction(-5, 7))))
    zero = FreeAlgebraPolynomial(alphabet=value.alphabet, terms=())

    self_bracket = commutator(value, value)
    zero_bracket = commutator(value, zero)
    assert self_bracket.commutator.terms == ()
    assert zero_bracket.commutator.terms == ()

    request = FreeAlgebraCommutatorRequest(left=value, right=zero)
    restored_request = FreeAlgebraCommutatorRequest.model_validate_json(
        request.model_dump_json()
    )
    result = TOOLS[0].run(restored_request)
    restored_result = FreeAlgebraCommutatorResult.model_validate_json(
        result.model_dump_json()
    )
    assert restored_result == result
    assert TOOLS[0].operation_id == "free_algebra.polynomial.commutator.compute"


def test_commutator_linear_fixture_distinguishes_word_order() -> None:
    x_plus_y = _polynomial(((("x",), Fraction(1)), (("y",), Fraction(1))))
    x_minus_y = _polynomial(((("x",), Fraction(1)), (("y",), Fraction(-1))))

    result = commutator(x_plus_y, x_minus_y).commutator

    assert _coefficient_map(result) == {
        ("y", "x"): Fraction(2),
        ("x", "y"): Fraction(-2),
    }


def test_commutator_rejects_candidate_output_before_convolution() -> None:
    dense = _dense_operand()

    with pytest.raises(OperationResourceAdmissionError) as caught:
        commutator(dense, dense)
    assert caught.value.errors()[0]["type"] == (
        "free_algebra.commutator.result_term_budget"
    )
