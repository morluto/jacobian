"""Exact contracts for bounded noncommutative polynomial powers."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.free_algebras import operations
from jacobian.math.free_algebras._models import (
    FreeAlgebraPolynomial,
    FreeAlgebraPolynomialPowerRequest,
    FreeAlgebraTerm,
    canonical_word_key,
)
from jacobian.math.free_algebras._tools import TOOLS
from jacobian.math.free_algebras.operations import multiply, power_polynomial

OPERATION_ID = "free_algebra.polynomial.power.compute"


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


def _coefficients(value: FreeAlgebraPolynomial) -> dict[tuple[str, ...], Fraction]:
    return {term.word: term.coefficient.as_fraction() for term in value.terms}


def test_zero_one_and_zero_polynomial_powers() -> None:
    alphabet = ("x", "y")
    polynomial = _polynomial(alphabet, {("x",): 2, ("y",): -1})
    zero = _polynomial(alphabet, {})

    assert _coefficients(power_polynomial(polynomial, 0)) == {(): Fraction(1)}
    assert power_polynomial(polynomial, 0).alphabet == alphabet
    assert power_polynomial(polynomial, 1) == polynomial
    assert power_polynomial(zero, 9) == zero
    assert power_polynomial(_polynomial((), {(): 3}), 0) == _polynomial((), {(): 1})


def test_binary_power_keeps_noncommuting_words_distinct() -> None:
    alphabet = ("x", "y")
    polynomial = _polynomial(alphabet, {("x",): 1, ("y",): 1})

    squared = power_polynomial(polynomial, 2)
    cubed = power_polynomial(polynomial, 3)

    assert _coefficients(squared) == {
        ("x", "x"): Fraction(1),
        ("x", "y"): Fraction(1),
        ("y", "x"): Fraction(1),
        ("y", "y"): Fraction(1),
    }
    assert _coefficients(cubed)[("x", "y", "x")] == 1
    assert _coefficients(cubed)[("y", "x", "x")] == 1
    assert ("x", "y", "x") != ("y", "x", "x")


def test_power_nine_keeps_internal_products_above_public_operand_limit() -> None:
    polynomial = _polynomial(("x", "y"), {("x",): 1, ("y",): 1})
    result = power_polynomial(polynomial, 9)
    assert len(result.terms) == 512


def test_product_allocation_admits_colliding_word_support() -> None:
    letter = "x" * 64
    polynomial = _polynomial(
        (letter,),
        {(letter,) * exponent: 1 for exponent in range(33)},
    )
    result = multiply(polynomial, polynomial).product
    assert len(result.terms) == 65


def _matrix_add(left, right):
    return tuple(
        tuple(left[row][col] + right[row][col] for col in range(2)) for row in range(2)
    )


def _matrix_multiply(left, right):
    return tuple(
        tuple(
            sum((left[row][k] * right[k][col] for k in range(2)), Fraction(0))
            for col in range(2)
        )
        for row in range(2)
    )


def _matrix_power(matrix, exponent: int):
    identity = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    result = identity
    for _ in range(exponent):
        result = _matrix_multiply(result, matrix)
    return result


def _evaluate(polynomial: FreeAlgebraPolynomial, matrices):
    identity = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    zero = ((Fraction(0), Fraction(0)), (Fraction(0), Fraction(0)))
    result = zero
    for term in polynomial.terms:
        value = identity
        for letter in term.word:
            value = _matrix_multiply(value, matrices[letter])
        scaled = tuple(
            tuple(entry * term.coefficient.as_fraction() for entry in row)
            for row in value
        )
        result = _matrix_add(result, scaled)
    return result


def test_power_agrees_with_independent_noncommutative_matrix_algebra() -> None:
    polynomial = _polynomial(("x", "y"), {("x",): 1, ("y",): 1})
    matrices = {
        "x": ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(0))),
        "y": ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))),
    }
    matrix_sum = _matrix_add(matrices["x"], matrices["y"])

    assert _evaluate(power_polynomial(polynomial, 5), matrices) == _matrix_power(
        matrix_sum, 5
    )


def test_exponent_limit_and_every_product_preflight_before_expansion(
    monkeypatch,
) -> None:
    polynomial = _polynomial(("x",), {("x",): 1})
    assert _coefficients(power_polynomial(polynomial, 64)) == {("x",) * 64: Fraction(1)}
    with pytest.raises(OperationResourceAdmissionError, match="exponent"):
        power_polynomial(polynomial, 65)
    with pytest.raises(OperationResourceAdmissionError, match="exponent"):
        power_polynomial(polynomial, True)
    with pytest.raises(ValidationError):
        FreeAlgebraPolynomialPowerRequest(polynomial=polynomial, exponent=65)

    alphabet = ("x", "y")
    terms = {
        tuple(alphabet[(index >> bit) & 1] for bit in range(6)): Fraction(10**31)
        for index in range(64)
    }
    coefficient_growth = _polynomial(alphabet, terms)
    calls = 0

    def unexpected_expansion(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("power convolution started before admission")

    monkeypatch.setattr(operations, "multiply_sparse", unexpected_expansion)
    with pytest.raises(OperationResourceAdmissionError):
        power_polynomial(coefficient_growth, 2)
    assert calls == 0


def test_power_operation_json_and_catalog_result_round_trip() -> None:
    tool = next(tool for tool in TOOLS if tool.operation_id == OPERATION_ID)
    request = FreeAlgebraPolynomialPowerRequest.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    result = tool.run(request)
    restored = FreeAlgebraPolynomial.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result
    assert _coefficients(restored) == {
        ("y", "y"): Fraction(1),
        ("y", "x"): Fraction(1),
        ("x", "y"): Fraction(1),
        ("x", "x"): Fraction(1),
    }
