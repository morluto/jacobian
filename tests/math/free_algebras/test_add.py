"""Exact contract tests for sparse noncommutative polynomial addition."""

from __future__ import annotations

from fractions import Fraction
from itertools import islice, product

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras import operations
from jacobian.math.free_algebras._models import (
    MAX_FREE_ALGEBRA_ADDITION_TERMS,
    MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS,
    MAX_FREE_ALGEBRA_OPERAND_TERMS,
    FreeAlgebraPolynomial,
    FreeAlgebraPolynomialAddRequest,
    FreeAlgebraPolynomialProductRequest,
    FreeAlgebraTerm,
    canonical_word_key,
)
from jacobian.math.free_algebras._tools import TOOLS
from jacobian.math.free_algebras.operations import add, multiply

OPERATION_ID = "free_algebra.polynomial.add.compute"


def _polynomial(
    alphabet: tuple[str, ...],
    coefficients: dict[tuple[str, ...], int | Fraction],
) -> FreeAlgebraPolynomial:
    ordered = sorted(
        coefficients.items(),
        key=lambda item: canonical_word_key(alphabet, item[0]),
        reverse=True,
    )
    return FreeAlgebraPolynomial(
        alphabet=alphabet,
        terms=tuple(
            FreeAlgebraTerm(
                coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
                word=word,
            )
            for word, coefficient in ordered
            if coefficient
        ),
    )


def _coefficient_map(
    value: FreeAlgebraPolynomial,
) -> dict[tuple[str, ...], Fraction]:
    return {term.word: term.coefficient.as_fraction() for term in value.terms}


def _dict_fraction_oracle(
    left: FreeAlgebraPolynomial,
    right: FreeAlgebraPolynomial,
) -> dict[tuple[str, ...], Fraction]:
    """Independent sparse-map definition of addition, including cancellation."""

    result: dict[tuple[str, ...], Fraction] = {}
    for polynomial in (left, right):
        for term in polynomial.terms:
            result[term.word] = result.get(term.word, Fraction(0)) + Fraction(
                term.coefficient.num, term.coefficient.den
            )
    return {word: coefficient for word, coefficient in result.items() if coefficient}


def _binary_words(alphabet: tuple[str, ...], count: int) -> tuple[tuple[str, ...], ...]:
    words: list[tuple[str, ...]] = []
    length = 0
    while len(words) < count:
        for word in product(alphabet, repeat=length):
            words.append(word)
            if len(words) == count:
                return tuple(words)
        length += 1
    return tuple(words)


def test_add_matches_independent_fraction_map_and_cancels_terms() -> None:
    alphabet = ("x", "y")
    left = _polynomial(
        alphabet,
        {("x",): Fraction(1, 3), ("y", "x"): 2, (): 5},
    )
    right = _polynomial(
        alphabet,
        {("x",): Fraction(-1, 3), ("x", "y"): Fraction(7, 4), (): -5},
    )

    result = add(left, right)

    assert _coefficient_map(result) == _dict_fraction_oracle(left, right)
    assert _coefficient_map(result) == {
        ("y", "x"): Fraction(2),
        ("x", "y"): Fraction(7, 4),
    }
    assert result.alphabet == alphabet
    assert tuple(term.word for term in result.terms) == tuple(
        sorted(
            (term.word for term in result.terms),
            key=lambda word: canonical_word_key(alphabet, word),
            reverse=True,
        )
    )


def test_zero_and_empty_alphabet_keep_their_parent() -> None:
    alphabet = ("x", "y")
    zero = _polynomial(alphabet, {})
    value = _polynomial(alphabet, {("x",): 3})
    assert add(zero, value) == value
    assert add(value, _polynomial(alphabet, {("x",): -3})).is_zero
    assert add(zero, zero).alphabet == alphabet

    scalar = _polynomial((), {(): Fraction(1, 3)})
    assert _coefficient_map(add(scalar, scalar)) == {(): Fraction(2, 3)}


def test_sum_of_disjoint_maximum_operands_admits_full_support() -> None:
    alphabet = ("x", "y")
    words = _binary_words(alphabet, 2 * MAX_FREE_ALGEBRA_OPERAND_TERMS)
    left_words = words[:MAX_FREE_ALGEBRA_OPERAND_TERMS]
    right_words = words[MAX_FREE_ALGEBRA_OPERAND_TERMS:]
    left = _polynomial(alphabet, dict.fromkeys(left_words, Fraction(2, 3)))
    right = _polynomial(alphabet, dict.fromkeys(right_words, Fraction(-3, 5)))

    result = add(left, right)

    assert len(result.terms) == MAX_FREE_ALGEBRA_ADDITION_TERMS
    assert _coefficient_map(result) == _dict_fraction_oracle(left, right)


def test_identical_ordered_alphabet_is_required() -> None:
    left = _polynomial(("x", "y"), {("x",): 1})
    right = _polynomial(("y", "x"), {("x",): 1})
    with pytest.raises(OperationDomainValidationError) as exc_info:
        add(left, right)
    assert exc_info.value.errors()[0]["type"] == "free_algebra.alphabet_mismatch"


def test_coefficient_growth_is_admitted_before_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alphabet = ("x",)
    boundary = 10**30
    result = add(
        _polynomial(alphabet, {("x",): boundary}),
        _polynomial(alphabet, {("x",): boundary}),
    )
    assert _coefficient_map(result) == {("x",): Fraction(2 * boundary)}
    assert len(str(2 * boundary)) <= MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS

    def aggregation_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("aggregation ran before growth admission")

    monkeypatch.setattr(operations, "add_sparse", aggregation_must_not_run)
    left = _polynomial(alphabet, {("x",): Fraction(1, 10**39 + 1)})
    right = _polynomial(alphabet, {("x",): Fraction(1, 10**39 + 3)})
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        add(left, right)
    assert exc_info.value.errors()[0]["type"] == (
        "free_algebra.addition_coefficient_growth_budget"
    )


def test_serialized_sum_composes_unchanged_with_multiplication_and_catalog() -> None:
    tool = next(command for command in TOOLS if command.operation_id == OPERATION_ID)
    request = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    assert isinstance(request, FreeAlgebraPolynomialAddRequest)
    native = add(request.left, request.right)
    public = tool.run(request)
    assert public == native
    round_trip = FreeAlgebraPolynomial.model_validate_json(native.model_dump_json())
    assert round_trip == native

    product_request = FreeAlgebraPolynomialProductRequest(
        left=round_trip,
        right=_polynomial(round_trip.alphabet, {("y",): 1}),
    )
    product = multiply(product_request.left, product_request.right).product
    assert _coefficient_map(product) == {
        ("x", "y", "y"): Fraction(3, 2),
        ("y", "y"): Fraction(1),
    }


def test_output_size_is_admitted_before_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alphabet = (chr(0x1D465) * 64, chr(0x1D466) * 64)
    words = tuple(
        islice(product(alphabet, repeat=32), 2 * MAX_FREE_ALGEBRA_OPERAND_TERMS)
    )
    left = _polynomial(alphabet, dict.fromkeys(words[:64], 1))
    right = _polynomial(alphabet, dict.fromkeys(words[64:], 1))

    def aggregation_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("aggregation ran before output admission")

    monkeypatch.setattr(operations, "add_sparse", aggregation_must_not_run)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        add(left, right)
    assert exc_info.value.errors()[0]["type"] == (
        "free_algebra.addition_output_cells_budget"
    )
