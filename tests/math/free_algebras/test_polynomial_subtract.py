"""Independent exact contract tests for noncommutative polynomial subtraction."""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import product

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.free_algebras._models import (
    MAX_FREE_ALGEBRA_ADDITION_TERMS,
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
    canonical_word_key,
)
from jacobian.math.free_algebras.polynomial_subtract import operations
from jacobian.math.free_algebras.polynomial_subtract.operations import subtract

OPERATION_ID = "free_algebra.polynomial.subtract.compute"


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


def _coefficient_map(value: FreeAlgebraPolynomial) -> dict[tuple[str, ...], Fraction]:
    return {term.word: term.coefficient.as_fraction() for term in value.terms}


def _difference_oracle(
    left: FreeAlgebraPolynomial,
    right: FreeAlgebraPolynomial,
) -> dict[tuple[str, ...], Fraction]:
    """Independent dictionary definition of the free-algebra difference."""

    result: dict[tuple[str, ...], Fraction] = {}
    for term in left.terms:
        result[term.word] = term.coefficient.as_fraction()
    for term in right.terms:
        result[term.word] = (
            result.get(term.word, Fraction(0)) - term.coefficient.as_fraction()
        )
    return {word: coefficient for word, coefficient in result.items() if coefficient}


def _words(alphabet: tuple[str, ...], count: int) -> tuple[tuple[str, ...], ...]:
    words: list[tuple[str, ...]] = []
    length = 0
    while len(words) < count:
        words.extend(product(alphabet, repeat=length))
        length += 1
    return tuple(words[:count])


def test_subtraction_matches_independent_rational_map_and_preserves_word_order() -> (
    None
):
    alphabet = ("x", "y")
    left = _polynomial(alphabet, {("x",): Fraction(5, 3), ("x", "y"): 2, (): 4})
    right = _polynomial(alphabet, {("x",): Fraction(2, 3), ("y", "x"): 2, (): 4})

    result = subtract(left, right)

    assert _coefficient_map(result) == _difference_oracle(left, right)
    assert _coefficient_map(result) == {
        ("x",): Fraction(1),
        ("x", "y"): Fraction(2),
        ("y", "x"): Fraction(-2),
    }
    assert result.alphabet == alphabet
    assert tuple(term.word for term in result.terms) == tuple(
        sorted(
            (term.word for term in result.terms),
            key=lambda word: canonical_word_key(alphabet, word),
            reverse=True,
        )
    )


def test_zero_empty_alphabet_and_exact_cancellation_keep_the_parent() -> None:
    zero = _polynomial(("x",), {})
    value = _polynomial(("x",), {("x",): Fraction(7, 5)})
    assert subtract(value, zero) == value
    assert subtract(value, value).is_zero
    assert subtract(zero, zero).alphabet == ("x",)

    scalar = _polynomial((), {(): Fraction(7, 5)})
    assert _coefficient_map(subtract(scalar, _polynomial((), {(): 2}))) == {
        (): Fraction(-3, 5)
    }


def test_same_sign_64_digit_collision_cancels_within_admission() -> None:
    value = _polynomial(("x",), {("x",): 10**63})
    assert subtract(value, value).is_zero


def test_overlapping_support_is_bounded_by_distinct_words() -> None:
    support = _words(("x", "y"), 65)
    value = _polynomial(("x", "y"), dict.fromkeys(support, Fraction(2, 3)))
    result = subtract(value, value)
    assert result.is_zero
    assert result.alphabet == value.alphabet


def test_identical_ordered_alphabet_is_required() -> None:
    left = _polynomial(("x", "y"), {("x",): 1})
    right = _polynomial(("y", "x"), {("x",): 1})
    with pytest.raises(OperationDomainValidationError) as error:
        subtract(left, right)
    assert error.value.errors()[0]["type"] == "free_algebra.alphabet_mismatch"


def test_common_denominator_factor_is_accounted_for_exactly() -> None:
    scale = 10**62
    left = _polynomial(("x",), {("x",): Fraction(1, 2 * scale)})
    right = _polynomial(("x",), {("x",): Fraction(1, 5 * scale)})

    result = subtract(left, right)

    assert _coefficient_map(result) == {("x",): Fraction(3, 10 * scale)}
    assert len(str(result.terms[0].coefficient.den)) == 64


def test_output_cell_estimate_sums_actual_word_lengths() -> None:
    long_word = ("x",) * 64
    words = (long_word,) + _words(("x", "y"), 35)
    value = _polynomial(("x", "y"), dict.fromkeys(words, 1))

    assert subtract(value, _polynomial(("x", "y"), {})) == value


def test_full_128_term_support_is_accepted_and_exact() -> None:
    support = _words(("x", "y"), MAX_FREE_ALGEBRA_ADDITION_TERMS)
    left = _polynomial(("x", "y"), dict.fromkeys(support[:64], Fraction(2, 3)))
    right = _polynomial(("x", "y"), dict.fromkeys(support[64:], Fraction(-3, 5)))

    result = subtract(left, right)

    assert len(result.terms) == MAX_FREE_ALGEBRA_ADDITION_TERMS
    assert _coefficient_map(result) == _difference_oracle(left, right)


def test_forged_noncanonical_polynomial_is_rejected_without_model_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coefficient = CanonicalRational.model_construct(num=2, den=2)
    term = FreeAlgebraTerm.model_construct(coefficient=coefficient, word=("x",))
    forged = FreeAlgebraPolynomial.model_construct(alphabet=("x",), terms=(term,))

    def unexpected_replay(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("subtraction replayed a source value through Pydantic")

    monkeypatch.setattr(FreeAlgebraPolynomial, "model_dump", unexpected_replay)
    with pytest.raises(OperationDomainValidationError) as error:
        subtract(forged, _polynomial(("x",), {}))
    assert error.value.errors()[0]["type"] == "free_algebra.polynomial_shape"


def test_forged_missing_coefficient_field_is_a_domain_error() -> None:
    term = FreeAlgebraTerm.model_construct(word=("x",))
    forged = FreeAlgebraPolynomial.model_construct(alphabet=("x",), terms=(term,))
    with pytest.raises(OperationDomainValidationError) as error:
        subtract(forged, _polynomial(("x",), {}))
    assert error.value.errors()[0]["type"] == "free_algebra.polynomial_shape"


def test_result_term_bound_rejects_before_sparse_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support = _words(("x", "y"), MAX_FREE_ALGEBRA_ADDITION_TERMS + 1)
    left = _polynomial(("x", "y"), dict.fromkeys(support[:64], 1))
    right = _polynomial(("x", "y"), dict.fromkeys(support[64:], 1))

    def unexpected_subtraction(
        *_args: object, **_kwargs: object
    ) -> FreeAlgebraPolynomial:
        raise AssertionError("sparse arithmetic ran before result admission")

    monkeypatch.setattr(operations, "_subtract_sparse", unexpected_subtraction)
    with pytest.raises(OperationResourceAdmissionError) as error:
        subtract(left, right)
    assert error.value.errors()[0]["type"] == (
        "free_algebra.subtraction_result_term_budget"
    )


def test_catalog_example_executes_and_round_trips_canonical_result() -> None:
    catalog = Catalog.open()
    tool = catalog.operation(OPERATION_ID)
    assert tool is not None
    response = invoke_operation(OPERATION_ID, tool.examples[0].input, catalog)
    result = tool.result_type.model_validate_json(json.dumps(response.output))
    assert isinstance(result, FreeAlgebraPolynomial)
    assert _coefficient_map(result) == {
        ("x",): Fraction(1),
        ("x", "y"): Fraction(3, 2),
        ("y",): Fraction(-1),
    }
