"""Mathematical tests for the free-algebra reversal anti-automorphism."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.free_algebras import operations
from jacobian.math.free_algebras._models import (
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
    canonical_word_key,
)
from jacobian.math.free_algebras._tools import TOOLS
from jacobian.math.free_algebras.operations import (
    add,
    multiply,
    reverse_polynomial_antiautomorphism,
)


def _polynomial(
    alphabet: tuple[str, ...],
    coefficients: dict[tuple[str, ...], int | Fraction],
) -> FreeAlgebraPolynomial:
    terms = tuple(
        FreeAlgebraTerm(
            coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
            word=word,
        )
        for word, coefficient in sorted(
            coefficients.items(),
            key=lambda item: canonical_word_key(alphabet, item[0]),
            reverse=True,
        )
        if coefficient
    )
    return FreeAlgebraPolynomial(alphabet=alphabet, terms=terms)


def _coefficient_map(value: FreeAlgebraPolynomial) -> dict[tuple[str, ...], Fraction]:
    return {
        term.word: Fraction(term.coefficient.num, term.coefficient.den)
        for term in value.terms
    }


def _product_oracle(
    left: FreeAlgebraPolynomial, right: FreeAlgebraPolynomial
) -> dict[tuple[str, ...], Fraction]:
    """Direct finite-word convolution independent of the production kernel."""

    result: dict[tuple[str, ...], Fraction] = {}
    for left_term in left.terms:
        for right_term in right.terms:
            word = left_term.word + right_term.word
            coefficient = Fraction(
                left_term.coefficient.num, left_term.coefficient.den
            ) * Fraction(right_term.coefficient.num, right_term.coefficient.den)
            result[word] = result.get(word, Fraction(0)) + coefficient
    return {word: value for word, value in result.items() if value}


def _reverse_oracle(
    coefficients: dict[tuple[str, ...], Fraction],
) -> dict[tuple[str, ...], Fraction]:
    return {tuple(reversed(word)): value for word, value in coefficients.items()}


def test_reverses_words_and_preserves_coefficients_alphabet_and_unit() -> None:
    source = _polynomial(
        ("x", "y"),
        {
            ("x", "y"): Fraction(2, 3),
            ("y", "x"): -5,
            ("x",): Fraction(7, 4),
            (): 9,
        },
    )

    result = reverse_polynomial_antiautomorphism(source)

    assert _coefficient_map(result) == {
        ("y", "x"): Fraction(2, 3),
        ("x", "y"): Fraction(-5),
        ("x",): Fraction(7, 4),
        (): Fraction(9),
    }
    assert result.alphabet == source.alphabet
    assert tuple(term.word for term in result.terms) == tuple(
        sorted(
            (term.word for term in result.terms),
            key=lambda word: canonical_word_key(result.alphabet, word),
            reverse=True,
        )
    )


def test_reversal_is_an_involution_and_round_trips_through_json() -> None:
    source = _polynomial(
        ("x", "y"),
        {("x", "y", "y"): Fraction(-11, 7), ("y", "x"): Fraction(5, 3)},
    )

    once = reverse_polynomial_antiautomorphism(source)
    decoded = FreeAlgebraPolynomial.model_validate_json(once.model_dump_json())
    twice = reverse_polynomial_antiautomorphism(decoded)

    assert decoded == once
    assert twice == source


def test_reversal_reverses_product_order_against_independent_word_oracle() -> None:
    left = _polynomial(("x", "y"), {("x",): 1, ("y",): Fraction(2, 3), (): -2})
    right = _polynomial(("x", "y"), {("x", "y"): 2, ("y",): -1, (): Fraction(3, 5)})
    left_times_right = multiply(left, right).product
    reversed_left = reverse_polynomial_antiautomorphism(left)
    reversed_right = reverse_polynomial_antiautomorphism(right)
    reverse_product = reverse_polynomial_antiautomorphism(left_times_right)
    reversed_right_times_left = multiply(reversed_right, reversed_left).product

    assert _coefficient_map(reverse_product) == _reverse_oracle(
        _product_oracle(left, right)
    )
    assert _coefficient_map(reversed_right_times_left) == _product_oracle(
        reversed_right, reversed_left
    )
    assert _coefficient_map(reverse_product) == _coefficient_map(
        reversed_right_times_left
    )


def test_output_admission_precedes_reversed_term_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _polynomial(("x",), {("x", "x"): 1})

    def construction_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("reversed terms were constructed before admission")

    monkeypatch.setattr(
        operations,
        "MAX_FREE_ALGEBRA_ANTIAUTOMORPHISM_OUTPUT_BYTES",
        1,
    )
    monkeypatch.setattr(operations, "FreeAlgebraTerm", construction_must_not_run)

    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        reverse_polynomial_antiautomorphism(value)

    assert exc_info.value.errors()[0]["type"] == (
        "free_algebra.antiautomorphism_output_budget"
    )


def test_catalog_example_is_a_direct_composable_polynomial_value() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id
        == "free_algebra.polynomial.reverse_antiautomorphism.compute"
    )
    request = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    result = tool.run(request)

    assert isinstance(result, FreeAlgebraPolynomial)
    assert result.alphabet == ("x", "y")
    assert _coefficient_map(result) == {
        ("y", "x"): Fraction(2),
        ("x", "y"): Fraction(3),
        ("x",): Fraction(4),
    }
    assert add(result, _polynomial(result.alphabet, {})) == result
