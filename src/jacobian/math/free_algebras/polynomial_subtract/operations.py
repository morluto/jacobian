"""Bounded exact subtraction for sparse noncommutative polynomials."""

from __future__ import annotations

from fractions import Fraction
from math import ceil, gcd, log2
from typing import NoReturn

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras._models import (
    MAX_FREE_ALGEBRA_ADDITION_OUTPUT_CELLS,
    MAX_FREE_ALGEBRA_ADDITION_TERMS,
    MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS,
    MAX_FREE_ALGEBRA_GENERATORS,
    MAX_FREE_ALGEBRA_LETTER_LENGTH,
    MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH,
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
)

MAX_FREE_ALGEBRA_SUBTRACTION_WORK = 1_500_000
_COEFFICIENT_LIMIT = 10**MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS


def _invalid(location: tuple[str | int, ...], message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=location,
        code="free_algebra.polynomial_shape",
        message=message,
    )


def _resource(code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("left", "right"),
        code=f"free_algebra.{code}",
        message=message,
    )


def _decimal_digits(value: int) -> int:
    if value.bit_length() > _COEFFICIENT_LIMIT.bit_length():
        return MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS + 1
    if abs(value) >= _COEFFICIENT_LIMIT:
        return MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS + 1
    return len(str(abs(value)))


def _validate_alphabet(
    alphabet: object,
    label: str,
) -> tuple[tuple[str, ...], dict[str, int], int]:
    if type(alphabet) is not tuple or len(alphabet) > MAX_FREE_ALGEBRA_GENERATORS:
        _invalid((label, "alphabet"), "generator alphabet exceeds its canonical bound")
    ranks: dict[str, int] = {}
    alphabet_scalars = 0
    for rank, letter in enumerate(alphabet):
        if (
            type(letter) is not str
            or not 1 <= len(letter) <= MAX_FREE_ALGEBRA_LETTER_LENGTH
            or any(0xD800 <= ord(character) <= 0xDFFF for character in letter)
            or letter in ranks
        ):
            _invalid((label, "alphabet", rank), "generator alphabet is not canonical")
        ranks[letter] = rank
        alphabet_scalars += len(letter)
    return alphabet, ranks, alphabet_scalars


def _validate_operand(
    value: FreeAlgebraPolynomial,
    label: str,
) -> tuple[dict[tuple[str, ...], Fraction], int, int, int, int]:
    """Check relied-upon structure in one bounded pass without model replay."""

    if type(value) is not FreeAlgebraPolynomial:
        _invalid((label,), "operand must be a canonical free-algebra polynomial")
    _alphabet, ranks, alphabet_scalars = _validate_alphabet(
        getattr(value, "alphabet", None), label
    )

    terms = getattr(value, "terms", None)
    if type(terms) is not tuple:
        _invalid((label, "terms"), "polynomial terms must be a canonical tuple")
    if len(terms) > MAX_FREE_ALGEBRA_ADDITION_TERMS:
        _resource(
            "subtraction_operand_term_budget",
            "subtraction operands exceed the admitted term count",
        )

    coefficients: dict[tuple[str, ...], Fraction] = {}
    maximum_word_scalars = 0
    maximum_coefficient_digits = 1
    previous_key: tuple[int, tuple[int, ...]] | None = None
    work = alphabet_scalars + len(terms)
    for index, term in enumerate(terms):
        location = (label, "terms", index)
        if type(term) is not FreeAlgebraTerm:
            _invalid(location, "polynomial support contains a noncanonical term")
        word = getattr(term, "word", None)
        if type(word) is not tuple or len(word) > MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH:
            _invalid((*location, "word"), "polynomial word exceeds its canonical bound")
        word_ranks: list[int] = []
        word_scalar_length = 0
        for position, letter in enumerate(word):
            if (
                type(letter) is not str
                or not 1 <= len(letter) <= MAX_FREE_ALGEBRA_LETTER_LENGTH
                or any(0xD800 <= ord(character) <= 0xDFFF for character in letter)
            ):
                _invalid(
                    (*location, "word", position),
                    "polynomial word contains a noncanonical generator label",
                )
            rank = ranks.get(letter)
            if rank is None:
                _invalid(
                    (*location, "word", position),
                    "polynomial word uses a letter outside its alphabet",
                )
            word_ranks.append(rank)
            word_scalar_length += len(letter)
        maximum_word_scalars = max(maximum_word_scalars, word_scalar_length)
        work += word_scalar_length + 4

        coefficient = getattr(term, "coefficient", None)
        if type(coefficient) is not CanonicalRational:
            _invalid((*location, "coefficient"), "term coefficient is not canonical")
        numerator = getattr(coefficient, "num", None)
        denominator = getattr(coefficient, "den", None)
        if (
            type(numerator) is not int
            or type(denominator) is not int
            or denominator <= 0
            or numerator == 0
        ):
            _invalid((*location, "coefficient"), "term coefficient is not canonical")
        digits = max(_decimal_digits(numerator), _decimal_digits(denominator))
        if digits > MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS:
            _resource(
                "subtraction_coefficient_digit_budget",
                "subtraction input coefficient exceeds its digit bound",
            )
        if gcd(abs(numerator), denominator) != 1:
            _invalid((*location, "coefficient"), "term coefficient is not canonical")
        maximum_coefficient_digits = max(maximum_coefficient_digits, digits)
        work += 2 * digits

        word_key = (len(word), tuple(word_ranks))
        if previous_key is not None and previous_key <= word_key:
            _invalid(
                location, "polynomial terms are duplicated or out of canonical order"
            )
        previous_key = word_key
        coefficients[word] = Fraction(numerator, denominator)

    return (
        coefficients,
        alphabet_scalars,
        maximum_word_scalars,
        maximum_coefficient_digits,
        work,
    )


def _subtraction_collision_digits(
    left: CanonicalRational, right: CanonicalRational
) -> int:
    left_num = _decimal_digits(left.num)
    right_num = _decimal_digits(right.num)
    left_den = _decimal_digits(left.den)
    right_den = _decimal_digits(right.den)
    if left.den == right.den:
        grows = (left.num < 0) != (right.num < 0)
        numerator_digits = max(left_num, right_num) + int(grows)
        denominator_digits = left_den
    else:
        numerator_digits = max(left_num + right_den, right_num + left_den) + 1
        denominator_digits = left_den + right_den
    return max(numerator_digits, denominator_digits)


def _subtract_sparse(
    alphabet: tuple[str, ...],
    left: dict[tuple[str, ...], Fraction],
    right: dict[tuple[str, ...], Fraction],
) -> FreeAlgebraPolynomial:
    """Compute an admitted exact sparse difference and canonicalize its support."""

    difference = dict(left)
    for word, coefficient in right.items():
        difference[word] = difference.get(word, Fraction(0)) - coefficient
    ranks = {letter: index for index, letter in enumerate(alphabet)}
    ordered = tuple(
        sorted(
            (
                (word, coefficient)
                for word, coefficient in difference.items()
                if coefficient
            ),
            key=lambda item: (
                len(item[0]),
                tuple(ranks[letter] for letter in item[0]),
            ),
            reverse=True,
        )
    )
    terms = tuple(
        FreeAlgebraTerm(
            coefficient=CanonicalRational.from_fraction(coefficient),
            word=word,
        )
        for word, coefficient in ordered
    )
    return FreeAlgebraPolynomial.model_construct(alphabet=alphabet, terms=terms)


def subtract(
    left: FreeAlgebraPolynomial,
    right: FreeAlgebraPolynomial,
) -> FreeAlgebraPolynomial:
    """Return the exact coefficientwise difference ``left - right``."""

    (
        left_coefficients,
        left_alphabet_scalars,
        max_word_scalars,
        left_digits,
        left_work,
    ) = _validate_operand(left, "left")
    (
        right_coefficients,
        _right_alphabet_scalars,
        right_word_scalars,
        right_digits,
        right_work,
    ) = _validate_operand(right, "right")
    if left.alphabet != right.alphabet:
        raise OperationDomainValidationError(
            location=("right", "alphabet"),
            code="free_algebra.alphabet_mismatch",
            message=(
                "both operands must be bound to the same ordered generator alphabet"
            ),
        )

    collision_digits = 1
    left_terms = {term.word: term for term in left.terms}
    right_terms = {term.word: term for term in right.terms}
    for word in left_terms.keys() & right_terms.keys():
        collision_digits = max(
            collision_digits,
            _subtraction_collision_digits(
                left_terms[word].coefficient,
                right_terms[word].coefficient,
            ),
        )
    predicted_coefficient_digits = max(left_digits, right_digits, collision_digits)
    if predicted_coefficient_digits > MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS:
        _resource(
            "subtraction_coefficient_growth_budget",
            "predicted difference coefficient growth exceeds its digit bound",
        )

    result_term_bound = len(left_coefficients.keys() | right_coefficients.keys())
    if result_term_bound > MAX_FREE_ALGEBRA_ADDITION_TERMS:
        _resource(
            "subtraction_result_term_budget",
            "difference support exceeds the admitted result term count",
        )
    predicted_output_cells = (
        left_alphabet_scalars
        + 64
        + result_term_bound
        * (
            64
            + max(max_word_scalars, right_word_scalars)
            + 2 * (predicted_coefficient_digits + 1)
        )
    )
    if predicted_output_cells > MAX_FREE_ALGEBRA_ADDITION_OUTPUT_CELLS:
        _resource(
            "subtraction_output_cells_budget",
            "predicted polynomial difference exceeds the output allocation bound",
        )
    work_bound = (
        left_work
        + right_work
        + result_term_bound
        * max(1, ceil(log2(max(2, result_term_bound))))
        * MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH
        + 8 * result_term_bound
        + 4 * result_term_bound * predicted_coefficient_digits
        + predicted_output_cells
    )
    if work_bound > MAX_FREE_ALGEBRA_SUBTRACTION_WORK:
        _resource(
            "subtraction_work_budget",
            "polynomial difference exceeds the admitted exact work bound",
        )

    return _subtract_sparse(left.alphabet, left_coefficients, right_coefficients)


__all__ = ["MAX_FREE_ALGEBRA_SUBTRACTION_WORK", "subtract"]
