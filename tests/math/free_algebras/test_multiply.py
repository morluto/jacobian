"""Exact contract tests for free associative polynomial multiplication."""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import product

import pytest
import sympy
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras import operations
from jacobian.math.free_algebras._models import (
    MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS,
    MAX_FREE_ALGEBRA_OPERAND_TERMS,
    MAX_FREE_ALGEBRA_RESULT_TERMS,
    MAX_FREE_ALGEBRA_WORD_LENGTH,
    FreeAlgebraPolynomial,
    FreeAlgebraPolynomialProductResult,
    FreeAlgebraTerm,
    canonical_word_key,
)
from jacobian.math.free_algebras._tools import TOOLS
from jacobian.math.free_algebras.operations import multiply

OPERATION_ID = "free_algebra.polynomial.multiply.compute"


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


def _coefficient_map(
    polynomial: FreeAlgebraPolynomial,
) -> dict[tuple[str, ...], Fraction]:
    return {term.word: term.coefficient.as_fraction() for term in polynomial.terms}


def _add_maps(
    *maps: dict[tuple[str, ...], Fraction],
) -> dict[tuple[str, ...], Fraction]:
    total: dict[tuple[str, ...], Fraction] = {}
    for mapping in maps:
        for word, value in mapping.items():
            total[word] = total.get(word, Fraction(0)) + value
    return {word: value for word, value in total.items() if value != 0}


def _fixed_length_words(
    alphabet: tuple[str, ...], length: int
) -> tuple[tuple[str, ...], ...]:
    return tuple(product(alphabet, repeat=length))


def _distinct_words(
    alphabet: tuple[str, ...], count: int
) -> tuple[tuple[str, ...], ...]:
    words: list[tuple[str, ...]] = []
    length = 0
    while len(words) < count:
        for word in product(alphabet, repeat=length):
            words.append(word)
            if len(words) == count:
                return tuple(words)
        length += 1
    return tuple(words)


def _sympy_expression(
    alphabet: tuple[str, ...], coefficients: dict[tuple[str, ...], Fraction]
) -> sympy.Expr:
    symbols = {letter: sympy.Symbol(letter, commutative=False) for letter in alphabet}
    total = sympy.Integer(0)
    for word, value in coefficients.items():
        monomial = sympy.Integer(1)
        for letter in word:
            monomial = monomial * symbols[letter]
        total += monomial * sympy.Rational(value.numerator, value.denominator)
    return sympy.expand(total)


def test_published_example_product_is_exact() -> None:
    alphabet = ("x", "y")
    left = _poly(alphabet, {("y",): 1, ("x",): 1})
    right = _poly(alphabet, {("y",): -1, ("x",): 1})
    result = multiply(left, right)

    assert _coefficient_map(result.product) == {
        ("y", "y"): Fraction(-1),
        ("y", "x"): Fraction(1),
        ("x", "y"): Fraction(-1),
        ("x", "x"): Fraction(1),
    }
    assert result.ledger.term_pair_count == 4
    assert result.ledger.collected_pair_count == 0
    assert result.ledger.result_term_count == 4
    assert result.product.alphabet == alphabet


def test_noncommutativity_keeps_xy_and_yx_distinct() -> None:
    alphabet = ("x", "y")
    x, y = _poly(alphabet, {("x",): 1}), _poly(alphabet, {("y",): 1})
    assert _coefficient_map(multiply(x, y).product) == {("x", "y"): Fraction(1)}
    assert _coefficient_map(multiply(y, x).product) == {("y", "x"): Fraction(1)}
    assert multiply(x, y).product != multiply(y, x).product


def test_like_words_collect_and_cancel() -> None:
    alphabet = ("x", "y")
    left = _poly(alphabet, {("x",): 2, ("y",): 1})
    right = _poly(alphabet, {("x",): 1, ("y",): -1})
    result = multiply(left, right)
    # (2x + y)(x - y) = 2xx - 2xy + yx - yy, no collision here.
    assert _coefficient_map(result.product) == {
        ("y", "y"): Fraction(-1),
        ("y", "x"): Fraction(1),
        ("x", "y"): Fraction(-2),
        ("x", "x"): Fraction(2),
    }
    assert result.ledger.distinct_product_word_count == 4
    assert result.ledger.collected_pair_count == 0

    # (xx - x)(y + xy) = xxxy + xxy - xy - xxy = xxxy - xy: the two
    # contributions to xxy collect and cancel exactly.
    collision = multiply(
        _poly(alphabet, {("x", "x"): 1, ("x",): -1}),
        _poly(alphabet, {("y",): 1, ("x", "y"): 1}),
    )
    assert _coefficient_map(collision.product) == {
        ("x", "x", "x", "y"): Fraction(1),
        ("x", "y"): Fraction(-1),
    }
    assert collision.ledger.term_pair_count == 4
    assert collision.ledger.distinct_product_word_count == 3
    assert collision.ledger.collected_pair_count == 1
    assert collision.ledger.zero_coefficient_word_count == 1
    assert collision.ledger.result_term_count == 2


def test_multiplication_is_associative_on_fixtures() -> None:
    alphabet = ("x", "y")
    a = _poly(alphabet, {("x",): 1, ("y",): 1})
    b = _poly(alphabet, {("x", "y"): 1, ("x",): -1})
    c = _poly(alphabet, {("y", "x"): 2, ("y",): 1})

    left_product = multiply(multiply(a, b).product, c).product
    right_product = multiply(a, multiply(b, c).product).product
    assert _coefficient_map(left_product) == _coefficient_map(right_product)


def test_distributivity_over_exact_addition() -> None:
    alphabet = ("x", "y")
    a = _poly(alphabet, {("x",): 2, ("y",): -1})
    b = _poly(alphabet, {("x",): 3, ("x", "y"): 1})
    c = _poly(alphabet, {("y",): 1, ("y", "y"): -2})

    left = multiply(
        a, _poly(alphabet, _add_maps(_coefficient_map(b), _coefficient_map(c)))
    )
    right = _add_maps(
        _coefficient_map(multiply(a, b).product),
        _coefficient_map(multiply(a, c).product),
    )
    assert _coefficient_map(left.product) == right


def test_empty_word_is_two_sided_identity() -> None:
    alphabet = ("x", "y")
    unit = _poly(alphabet, {(): 1})
    value = _poly(alphabet, {("y", "x"): Fraction(3, 4), ("x",): -2})
    assert multiply(value, unit).product == value
    assert multiply(unit, value).product == value
    assert multiply(unit, unit).product == unit


def test_degree_is_additive_on_monomials() -> None:
    alphabet = ("x", "y")
    left = _poly(alphabet, {("x", "x", "y"): 1})
    right = _poly(alphabet, {("y", "x"): 1})
    product_result = multiply(left, right).product
    assert tuple(term.word for term in product_result.terms) == (
        ("x", "x", "y", "y", "x"),
    )
    assert product_result.terms[0].word == ("x", "x", "y", "y", "x")


def test_zero_polynomial_and_empty_alphabet_boundaries() -> None:
    alphabet = ("x", "y")
    zero = _poly(alphabet, {})
    value = _poly(alphabet, {("x",): 1})
    assert multiply(zero, value).product.is_zero
    assert multiply(value, zero).product.is_zero
    result = multiply(zero, zero)
    assert result.product.is_zero
    assert result.ledger.term_pair_count == 0
    assert result.ledger.result_term_count == 0

    empty_unit = _poly((), {(): 3})
    assert _coefficient_map(multiply(empty_unit, _poly((), {(): 5})).product) == {
        (): Fraction(15)
    }


def test_alphabet_mismatch_is_a_domain_rejection() -> None:
    left = _poly(("x",), {("x",): 1})
    right = _poly(("x", "y"), {("x",): 1})
    with pytest.raises(OperationDomainValidationError) as exc_info:
        multiply(left, right)
    assert exc_info.value.errors()[0]["type"] == "free_algebra.alphabet_mismatch"


def test_result_round_trips_and_matches_native_and_catalog() -> None:
    request_type = TOOLS[0].request_type
    request = request_type.model_validate_json(
        encode_strict_json(TOOLS[0].examples[0].input), strict=True
    )
    native = multiply(request.left, request.right)
    assert (
        FreeAlgebraPolynomialProductResult.model_validate_json(native.model_dump_json())
        == native
    )

    command = next(tool for tool in TOOLS if tool.operation_id == OPERATION_ID)
    public = command.run(request)
    assert public.model_dump(mode="json") == native.model_dump(mode="json")


def test_cross_check_against_sympy_noncommutative_symbols() -> None:
    alphabet = ("x", "y", "z")
    left_coefficients = {
        ("x", "y"): Fraction(2),
        ("z",): Fraction(-1, 3),
        (): Fraction(5),
    }
    right_coefficients = {
        ("y", "z"): Fraction(1, 2),
        ("x",): Fraction(3),
        (): Fraction(-2),
    }
    left, right = (
        _poly(alphabet, left_coefficients),
        _poly(alphabet, right_coefficients),
    )
    result = multiply(left, right)

    expected = sympy.expand(
        _sympy_expression(alphabet, left_coefficients)
        * _sympy_expression(alphabet, right_coefficients)
    )
    actual = _sympy_expression(alphabet, _coefficient_map(result.product))
    assert sympy.expand(actual - expected) == 0

    # Reversed order must differ for noncommuting words.
    reversed_expected = sympy.expand(
        _sympy_expression(alphabet, right_coefficients)
        * _sympy_expression(alphabet, left_coefficients)
    )
    reversed_actual = _sympy_expression(
        alphabet, _coefficient_map(multiply(right, left).product)
    )
    assert sympy.expand(reversed_actual - reversed_expected) == 0
    assert sympy.expand(actual - reversed_expected) != 0


def test_operand_term_boundary_is_enforced_before_expansion() -> None:
    alphabet = ("x", "y")
    accepted = _poly(
        alphabet,
        dict.fromkeys(_distinct_words(alphabet, MAX_FREE_ALGEBRA_OPERAND_TERMS), 1),
    )
    single = _poly(alphabet, {("x",): 1})
    result = multiply(accepted, single)
    assert result.ledger.left_term_count == MAX_FREE_ALGEBRA_OPERAND_TERMS

    over = _poly(
        alphabet,
        dict.fromkeys(_distinct_words(alphabet, MAX_FREE_ALGEBRA_OPERAND_TERMS + 1), 1),
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        multiply(over, single)
    assert exc_info.value.errors()[0]["type"] == "free_algebra.operand_term_budget"


def test_operand_word_length_boundary_is_enforced() -> None:
    alphabet = ("x", "y")
    boundary = _poly(alphabet, {("x",) * MAX_FREE_ALGEBRA_WORD_LENGTH: 1})
    right = _poly(alphabet, {("y",): 1})
    assert len(multiply(boundary, right).product.terms[0].word) == (
        MAX_FREE_ALGEBRA_WORD_LENGTH + 1
    )

    over = _poly(alphabet, {("x",) * (MAX_FREE_ALGEBRA_WORD_LENGTH + 1): 1})
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        multiply(over, right)
    assert exc_info.value.errors()[0]["type"] == (
        "free_algebra.operand_word_length_budget"
    )


def test_coefficient_growth_is_preflighted() -> None:
    alphabet = ("x",)
    boundary = _poly(alphabet, {("x",): 10**31})
    result = multiply(boundary, _poly(alphabet, {("x",): 1}))
    assert result.product.terms[0].coefficient.as_fraction() == 10**31

    pairwise = multiply(
        _poly(alphabet, {("x",): 10**31}), _poly(alphabet, {("x",): 10**31})
    )
    assert pairwise.ledger.max_result_coefficient_digits <= (
        MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS
    )

    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        multiply(_poly(alphabet, {("x",): 10**32}), _poly(alphabet, {("x",): 10**32}))
    assert exc_info.value.errors()[0]["type"] == (
        "free_algebra.coefficient_growth_budget"
    )


def test_full_result_term_budget_is_reachable_and_ledger_is_exact() -> None:
    alphabet = ("x", "y")
    words = _fixed_length_words(alphabet, 6)
    assert len(words) == 64
    left = _poly(alphabet, dict.fromkeys(words, 1))
    right = _poly(alphabet, dict.fromkeys(words, 1))
    result = multiply(left, right)

    assert result.ledger.term_pair_count == MAX_FREE_ALGEBRA_RESULT_TERMS
    assert result.ledger.distinct_product_word_count == MAX_FREE_ALGEBRA_RESULT_TERMS
    assert result.ledger.collected_pair_count == 0
    assert result.ledger.result_term_count == MAX_FREE_ALGEBRA_RESULT_TERMS
    assert len(result.product.terms) == MAX_FREE_ALGEBRA_RESULT_TERMS
    assert result.ledger.max_result_word_length == 12


def test_product_output_allocation_is_admitted_before_expansion(monkeypatch) -> None:
    alphabet = ("a" * 64, "b" * 64)
    words = tuple(
        tuple(alphabet[(index >> bit) & 1] for bit in range(6))
        + (alphabet[0],) * (MAX_FREE_ALGEBRA_WORD_LENGTH - 6)
        for index in range(MAX_FREE_ALGEBRA_OPERAND_TERMS)
    )
    polynomial = _poly(alphabet, dict.fromkeys(words, 1))

    def unexpected_expansion(*args, **kwargs):
        raise AssertionError("product expansion started before output admission")

    monkeypatch.setattr(operations, "multiply_sparse", unexpected_expansion)
    with pytest.raises(OperationResourceAdmissionError, match="allocation bound"):
        multiply(polynomial, polynomial)


def test_tampered_result_structures_are_rejected() -> None:
    alphabet = ("x", "y")
    result = multiply(_poly(alphabet, {("x",): 1}), _poly(alphabet, {("y",): 1}))
    payload = result.model_dump(mode="json")

    bad_ledger = {**payload, "ledger": {**payload["ledger"], "result_term_count": 2}}
    with pytest.raises(ValidationError):
        FreeAlgebraPolynomialProductResult.model_validate_json(json.dumps(bad_ledger))

    bad_word = {
        **payload,
        "product": {
            "alphabet": ["x", "y"],
            "terms": [{"coefficient": {"num": "1", "den": "1"}, "word": ["z"]}],
        },
    }
    with pytest.raises(ValidationError):
        FreeAlgebraPolynomialProductResult.model_validate_json(json.dumps(bad_word))

    mismatched_alphabet = {
        **payload,
        "product": {**payload["product"], "alphabet": ["x", "y", "z"]},
    }
    with pytest.raises(ValidationError):
        FreeAlgebraPolynomialProductResult.model_validate_json(
            json.dumps(mismatched_alphabet)
        )
