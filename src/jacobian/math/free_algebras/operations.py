"""Native exact free associative word, polynomial, and ideal operations.

Each operation performs its semantic admission before its exact kernel and
constructs the canonical result without replaying the computed mathematics.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from fractions import Fraction
from itertools import product
from math import factorial, gcd, lcm
from typing import Any, Literal, cast

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras._kernel import add_sparse, multiply_sparse
from jacobian.math.free_algebras._models import (
    MAX_FREE_ALGEBRA_ADDITION_OUTPUT_BYTES,
    MAX_FREE_ALGEBRA_ADDITION_TERMS,
    MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS,
    MAX_FREE_ALGEBRA_GS_COMPOSITIONS,
    MAX_FREE_ALGEBRA_GS_PAIR_CHECKS,
    MAX_FREE_ALGEBRA_GS_REDUCTION_STEPS,
    MAX_FREE_ALGEBRA_IDEAL_COMPONENT_CONTEXTS,
    MAX_FREE_ALGEBRA_IDEAL_COMPONENT_MATRIX_CELLS,
    MAX_FREE_ALGEBRA_IDEAL_COMPONENT_SERIALIZED_BYTES,
    MAX_FREE_ALGEBRA_IDEAL_COMPONENT_WORDS,
    MAX_FREE_ALGEBRA_IDEAL_PREFIX_BASIS,
    MAX_FREE_ALGEBRA_IDEAL_PREFIX_SERIALIZED_BYTES,
    MAX_FREE_ALGEBRA_IDEAL_PREFIX_TOTAL_TERMS,
    MAX_FREE_ALGEBRA_LETTER_LENGTH,
    MAX_FREE_ALGEBRA_OPERAND_TERMS,
    MAX_FREE_ALGEBRA_QUOTIENT_PROFILE_CANDIDATES,
    MAX_FREE_ALGEBRA_QUOTIENT_PROFILE_OUTPUT_BYTES,
    MAX_FREE_ALGEBRA_RESULT_TERMS,
    MAX_FREE_ALGEBRA_SUBSTITUTION_EXPANSIONS,
    MAX_FREE_ALGEBRA_SUBSTITUTION_OUTPUT_BYTES,
    MAX_FREE_ALGEBRA_SUBSTITUTION_WORK,
    MAX_FREE_ALGEBRA_TERM_PAIRS,
    MAX_FREE_ALGEBRA_WORD_LENGTH,
    MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH,
    MAX_FREE_WORD_FACTOR_DISTINCT,
    MAX_FREE_WORD_FACTOR_LETTER_CELLS,
    MAX_FREE_WORD_FACTOR_OCCURRENCES,
    MAX_FREE_WORD_OVERLAP_ALIGNMENTS,
    MAX_FREE_WORD_OVERLAP_LETTER_CELLS,
    MAX_FREE_WORD_OVERLAP_WORK,
    MAX_FREE_WORD_POWER_EXPONENT,
    MAX_FREE_WORD_SPLIT_LETTER_CELLS,
    MAX_FREE_WORD_SPLITS,
    FreeAlgebraIdeal,
    FreeAlgebraIdealDegreeComponentResult,
    FreeAlgebraIdealMembershipResult,
    FreeAlgebraIdealPrefixResult,
    FreeAlgebraPolynomial,
    FreeAlgebraPolynomialHomomorphism,
    FreeAlgebraPolynomialProductResult,
    FreeAlgebraQuotientDegreeComponent,
    FreeAlgebraQuotientProfileResult,
    FreeAlgebraTerm,
    FreeAlgebraWord,
    FreeAlgebraWordCompareResult,
    FreeAlgebraWordFactorOccurrences,
    FreeAlgebraWordFactorsResult,
    FreeAlgebraWordOverlapResult,
    FreeAlgebraWordOverlapWitness,
    FreeAlgebraWordPairResult,
    FreeAlgebraWordPowerResult,
    FreeAlgebraWordPrefixesResult,
    FreeAlgebraWordPrefixSplit,
    FreeAlgebraWordReverseResult,
    FreeAlgebraWordSubstitution,
    FreeAlgebraWordSubstitutionResult,
    FreeAlgebraWordSuffixesResult,
    FreeAlgebraWordSuffixSplit,
    FreeWordImageInterval,
    GroebnerShirshovResult,
    canonical_word_key,
)


def _reject_resource(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=location,
        code=f"free_algebra.{code}",
        message=message,
    )


def _admit_word(value: FreeAlgebraWord, *, label: str) -> FreeAlgebraWord:
    try:
        admitted = FreeAlgebraWord.model_validate(value.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=(label,),
            code="free_algebra.word_shape",
            message="the free-algebra word is not canonical",
        ) from exc
    if admitted.length > MAX_FREE_ALGEBRA_WORD_LENGTH:
        raise OperationDomainValidationError(
            location=(label, "letters"),
            code="free_algebra.word_source_length",
            message="source words may contain at most 32 letters",
        )
    return admitted


def _admit_word_pair(
    left: FreeAlgebraWord, right: FreeAlgebraWord
) -> tuple[FreeAlgebraWord, FreeAlgebraWord]:
    left_value = _admit_word(left, label="left")
    right_value = _admit_word(right, label="right")
    if left_value.alphabet != right_value.alphabet:
        raise OperationDomainValidationError(
            location=("right", "alphabet"),
            code="free_algebra.word_alphabet_mismatch",
            message="both words must use the same ordered generator alphabet",
        )
    return left_value, right_value


def _admit_polynomial(
    value: FreeAlgebraPolynomial, *, label: str
) -> FreeAlgebraPolynomial:
    try:
        return FreeAlgebraPolynomial.model_validate(value.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=(label,),
            code="free_algebra.polynomial_shape",
            message="the free-algebra polynomial is not canonical",
        ) from exc


def _admit_product(
    left: FreeAlgebraPolynomial, right: FreeAlgebraPolynomial
) -> tuple[FreeAlgebraPolynomial, FreeAlgebraPolynomial]:
    """Admit one product request before any word-pair expansion.

    The structural value contracts already established alphabet distinctness,
    canonical support, and per-coefficient digit bounds.  This shared
    admission helper adds the product operation's envelope: alphabet identity,
    operand term and word-length budgets, the term-pair product count, the
    result term count, and coefficient growth.
    """

    left = _admit_polynomial(left, label="left")
    right = _admit_polynomial(right, label="right")
    if left.alphabet != right.alphabet:
        raise OperationDomainValidationError(
            location=("right", "alphabet"),
            code="free_algebra.alphabet_mismatch",
            message=(
                "both operands must be bound to the same ordered generator alphabet"
            ),
        )

    max_operand_digits = 0
    for side, polynomial in (("left", left), ("right", right)):
        if len(polynomial.terms) > MAX_FREE_ALGEBRA_OPERAND_TERMS:
            _reject_resource(
                (side, "terms"),
                "operand_term_budget",
                f"{side} operand exceeds the "
                f"{MAX_FREE_ALGEBRA_OPERAND_TERMS}-term multiplication budget",
            )
        for index, term in enumerate(polynomial.terms):
            if len(term.word) > MAX_FREE_ALGEBRA_WORD_LENGTH:
                _reject_resource(
                    (side, "terms", index, "word"),
                    "operand_word_length_budget",
                    f"{side} operand word exceeds the "
                    f"{MAX_FREE_ALGEBRA_WORD_LENGTH}-letter multiplication "
                    "budget",
                )
            max_operand_digits = max(
                max_operand_digits,
                canonical_rational_component_digits(term.coefficient),
            )

    term_pair_count = len(left.terms) * len(right.terms)
    if term_pair_count > MAX_FREE_ALGEBRA_TERM_PAIRS:
        _reject_resource(
            ("left", "terms"),
            "term_pair_budget",
            "product term pairs exceed the "
            f"{MAX_FREE_ALGEBRA_TERM_PAIRS}-pair multiplication budget",
        )
    # Distinct product words are a subset of the term pairs, so this bounds
    # the result term count before product expansion.
    if term_pair_count > MAX_FREE_ALGEBRA_RESULT_TERMS:
        _reject_resource(
            ("left", "terms"),
            "result_term_budget",
            "product can exceed the "
            f"{MAX_FREE_ALGEBRA_RESULT_TERMS}-term result budget",
        )

    addition_digits = len(str(term_pair_count - 1)) if term_pair_count >= 2 else 0
    predicted_coefficient_digits = 2 * max_operand_digits + addition_digits
    if predicted_coefficient_digits > MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS:
        _reject_resource(
            ("left", "terms"),
            "coefficient_growth_budget",
            "predicted product coefficient growth exceeds the "
            f"{MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS}-digit multiplication budget",
        )
    return left, right


def _sum_coefficient_digit_bound(
    left: CanonicalRational, right: CanonicalRational
) -> int:
    """Bound canonical component digits of a rational sum without adding it."""

    left_num = len(str(abs(left.num)))
    right_num = len(str(abs(right.num)))
    left_den = len(str(left.den))
    right_den = len(str(right.den))
    if left.den == right.den:
        if (left.num < 0) != (right.num < 0):
            numerator_digits = max(left_num, right_num)
        else:
            numerator_digits = max(left_num, right_num) + 1
        denominator_digits = left_den
    else:
        numerator_digits = max(left_num + right_den, right_num + left_den) + 1
        denominator_digits = left_den + right_den
    return max(numerator_digits, denominator_digits)


def _admit_add(
    left: FreeAlgebraPolynomial, right: FreeAlgebraPolynomial
) -> tuple[FreeAlgebraPolynomial, FreeAlgebraPolynomial]:
    """Admit sparse addition before coefficient aggregation."""

    left = _admit_polynomial(left, label="left")
    right = _admit_polynomial(right, label="right")
    if left.alphabet != right.alphabet:
        raise OperationDomainValidationError(
            location=("right", "alphabet"),
            code="free_algebra.alphabet_mismatch",
            message=(
                "both operands must be bound to the same ordered generator alphabet"
            ),
        )

    for side, polynomial in (("left", left), ("right", right)):
        if len(polynomial.terms) > MAX_FREE_ALGEBRA_OPERAND_TERMS:
            _reject_resource(
                (side, "terms"),
                "addition_operand_term_budget",
                f"{side} operand exceeds the "
                f"{MAX_FREE_ALGEBRA_OPERAND_TERMS}-term addition budget",
            )
        for index, term in enumerate(polynomial.terms):
            if len(term.word) > MAX_FREE_ALGEBRA_WORD_LENGTH:
                _reject_resource(
                    (side, "terms", index, "word"),
                    "addition_operand_word_length_budget",
                    "addition operand words are limited to "
                    f"{MAX_FREE_ALGEBRA_WORD_LENGTH} letters",
                )
    left_by_word = {term.word: term for term in left.terms}
    right_by_word = {term.word: term for term in right.terms}
    maximum_input_digits = max(
        (
            canonical_rational_component_digits(term.coefficient)
            for term in (*left.terms, *right.terms)
        ),
        default=1,
    )
    collision_digit_bound = max(
        (
            _sum_coefficient_digit_bound(
                left_by_word[word].coefficient,
                right_by_word[word].coefficient,
            )
            for word in left_by_word.keys() & right_by_word.keys()
        ),
        default=1,
    )
    predicted_coefficient_digits = max(maximum_input_digits, collision_digit_bound)
    if predicted_coefficient_digits > MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS:
        _reject_resource(
            ("left", "right"),
            "addition_coefficient_growth_budget",
            "predicted sum coefficient growth exceeds the "
            f"{MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS}-digit bound",
        )

    result_term_bound = len(left.terms) + len(right.terms)
    if result_term_bound > MAX_FREE_ALGEBRA_ADDITION_TERMS:
        _reject_resource(
            ("left", "right", "terms"),
            "addition_result_term_budget",
            "sum support exceeds the admitted "
            f"{MAX_FREE_ALGEBRA_ADDITION_TERMS}-term bound",
        )

    alphabet_bytes = len(
        json.dumps(left.alphabet, ensure_ascii=True, separators=(",", ":"))
    )
    maximum_word_bytes = max(
        (
            len(json.dumps(term.word, ensure_ascii=True, separators=(",", ":")))
            for term in (*left.terms, *right.terms)
        ),
        default=2,
    )
    # The fixed term allowance covers object keys, brackets, commas, signs,
    # and JSON quoting. Each output word is drawn unchanged from an input.
    predicted_output_bytes = (
        alphabet_bytes
        + 64
        + result_term_bound
        * (64 + maximum_word_bytes + 2 * (predicted_coefficient_digits + 1))
    )
    if predicted_output_bytes > MAX_FREE_ALGEBRA_ADDITION_OUTPUT_BYTES:
        _reject_resource(
            ("left", "right"),
            "addition_output_bytes_budget",
            "predicted serialized polynomial sum exceeds the "
            f"{MAX_FREE_ALGEBRA_ADDITION_OUTPUT_BYTES}-byte bound",
        )
    return left, right


def add(
    left: FreeAlgebraPolynomial, right: FreeAlgebraPolynomial
) -> FreeAlgebraPolynomial:
    """Add two sparse noncommutative polynomials exactly."""

    left, right = _admit_add(left, right)
    return add_sparse(left, right)


def multiply(
    left: FreeAlgebraPolynomial, right: FreeAlgebraPolynomial
) -> FreeAlgebraPolynomialProductResult:
    """Multiply two sparse noncommutative polynomials exactly."""

    left, right = _admit_product(left, right)
    product, ledger = multiply_sparse(left, right)
    return FreeAlgebraPolynomialProductResult.model_construct(
        left=left,
        right=right,
        product=product,
        ledger=ledger,
    )


def _admit_substitution_images(
    substitution: FreeAlgebraPolynomialHomomorphism,
) -> tuple[dict[str, tuple[FreeAlgebraTerm, ...]], dict[str, int], dict[str, int], int]:
    """Reauthenticate images and gather the bounded expansion metadata."""
    image_term_counts: dict[str, int] = {}
    image_word_lengths: dict[str, int] = {}
    image_terms: dict[str, tuple[FreeAlgebraTerm, ...]] = {}
    maximum_encoded_letter_bytes = max(
        (
            len(json.dumps(letter, ensure_ascii=True))
            for letter in substitution.target_alphabet
        ),
        default=2,
    )
    for letter, image in zip(
        substitution.source_alphabet,
        substitution.images,
        strict=True,
    ):
        if len(image.terms) > MAX_FREE_ALGEBRA_OPERAND_TERMS:
            _reject_resource(
                ("substitution", "images", letter, "terms"),
                "substitution_image_term_budget",
                "each generator image is limited to 64 terms",
            )
        image_term_counts[letter] = len(image.terms)
        image_terms[letter] = image.terms
        image_word_lengths[letter] = max(
            (len(term.word) for term in image.terms), default=0
        )
        for index, term in enumerate(image.terms):
            if len(term.word) > MAX_FREE_ALGEBRA_WORD_LENGTH:
                _reject_resource(
                    ("substitution", "images", letter, "terms", index, "word"),
                    "substitution_image_word_budget",
                    "generator image words are limited to 32 letters",
                )
    return (
        image_terms,
        image_term_counts,
        image_word_lengths,
        maximum_encoded_letter_bytes,
    )


def _preflight_substitution_expansion(
    substitution: FreeAlgebraPolynomialHomomorphism,
    source: FreeAlgebraPolynomial,
    image_terms: dict[str, tuple[FreeAlgebraTerm, ...]],
    image_term_counts: dict[str, int],
    image_word_lengths: dict[str, int],
    maximum_encoded_letter_bytes: int,
) -> None:
    """Admit expansion, coefficient, work, and result envelopes up front."""

    expansion_count = 0
    maximum_source_word_length = 0
    maximum_output_word_length = 0
    maximum_contribution_digits = 0
    for term in source.terms:
        maximum_source_word_length = max(maximum_source_word_length, len(term.word))
        expansion = 1
        output_length = 0
        contribution_digits = canonical_rational_component_digits(term.coefficient)
        for letter in term.word:
            expansion *= image_term_counts[letter]
            if not expansion:
                break
            output_length += image_word_lengths[letter]
            contribution_digits += max(
                (
                    canonical_rational_component_digits(image_term.coefficient)
                    for image_term in image_terms[letter]
                ),
                default=0,
            )
        expansion_count += expansion
        maximum_output_word_length = max(maximum_output_word_length, output_length)
        maximum_contribution_digits = max(
            maximum_contribution_digits, contribution_digits
        )
        if expansion_count > MAX_FREE_ALGEBRA_SUBSTITUTION_EXPANSIONS:
            _reject_resource(
                ("polynomial", "terms"),
                "substitution_expansion_budget",
                "polynomial substitution exceeds the admitted expansion count",
            )

    if maximum_output_word_length > MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH:
        _reject_resource(
            ("substitution",),
            "substitution_word_length_budget",
            "substituted words exceed the 64-letter exact value limit",
        )
    sort_comparison_bound = max(1, expansion_count.bit_length())
    work_bound = expansion_count * (
        1
        + maximum_source_word_length
        + maximum_output_word_length * (1 + sort_comparison_bound)
    )
    if work_bound > MAX_FREE_ALGEBRA_SUBSTITUTION_WORK:
        _reject_resource(
            ("polynomial", "terms"),
            "substitution_work_budget",
            "polynomial substitution exceeds the admitted exact work bound",
        )
    if expansion_count > MAX_FREE_ALGEBRA_RESULT_TERMS:
        _reject_resource(
            ("polynomial", "terms"),
            "substitution_result_term_budget",
            "polynomial substitution can exceed the exact result term limit",
        )

    predicted_coefficient_digits = expansion_count * maximum_contribution_digits + (
        len(str(expansion_count)) if expansion_count > 1 else 0
    )
    if predicted_coefficient_digits > MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS:
        _reject_resource(
            ("substitution",),
            "substitution_coefficient_growth",
            "predicted exact coefficient growth exceeds the 64-digit limit",
        )
    output_byte_bound = expansion_count * (
        128
        + maximum_output_word_length * (maximum_encoded_letter_bytes + 4)
        + 2 * MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS
    )
    if output_byte_bound > MAX_FREE_ALGEBRA_SUBSTITUTION_OUTPUT_BYTES:
        _reject_resource(
            ("polynomial",),
            "substitution_output_bytes",
            "predicted canonical output exceeds the 2 MB limit",
        )


def _expand_polynomial_substitution(
    substitution: FreeAlgebraPolynomialHomomorphism,
    source: FreeAlgebraPolynomial,
    image_terms: dict[str, tuple[FreeAlgebraTerm, ...]],
) -> FreeAlgebraPolynomial:
    accumulated: dict[tuple[str, ...], Fraction] = {}
    for term in source.terms:
        factors = tuple(image_terms[letter] for letter in term.word)
        for choices in product(*factors):
            word_parts: list[str] = []
            coefficient = term.coefficient.as_fraction()
            for choice in choices:
                word_parts.extend(choice.word)
                coefficient *= choice.coefficient.as_fraction()
            word = tuple(word_parts)
            accumulated[word] = accumulated.get(word, Fraction(0)) + coefficient

    ordered = tuple(
        sorted(
            (
                (word, coefficient)
                for word, coefficient in accumulated.items()
                if coefficient
            ),
            key=lambda item: canonical_word_key(substitution.target_alphabet, item[0]),
            reverse=True,
        )
    )
    return FreeAlgebraPolynomial.model_construct(
        alphabet=substitution.target_alphabet,
        terms=tuple(
            FreeAlgebraTerm(
                coefficient=CanonicalRational.from_fraction(coefficient), word=word
            )
            for word, coefficient in ordered
        ),
    )


def substitute_polynomial(
    substitution: FreeAlgebraPolynomialHomomorphism,
    polynomial: FreeAlgebraPolynomial,
) -> FreeAlgebraPolynomial:
    """Apply the exact unital QQ-algebra map specified on free generators."""

    try:
        canonical_substitution = FreeAlgebraPolynomialHomomorphism.model_validate(
            substitution.model_dump()
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("substitution",),
            code="free_algebra.polynomial_substitution_shape",
            message="the polynomial homomorphism is not canonical",
        ) from exc
    source = _admit_polynomial(polynomial, label="polynomial")
    if source.alphabet != canonical_substitution.source_alphabet:
        raise OperationDomainValidationError(
            location=("polynomial", "alphabet"),
            code="free_algebra.polynomial_substitution_source_alphabet",
            message="polynomial must use the substitution source alphabet",
        )
    if len(source.terms) > MAX_FREE_ALGEBRA_OPERAND_TERMS:
        _reject_resource(
            ("polynomial", "terms"),
            "substitution_operand_term_budget",
            "source polynomial exceeds the 64-term substitution budget",
        )
    images, counts, lengths, encoded_letter_bytes = _admit_substitution_images(
        canonical_substitution
    )
    _preflight_substitution_expansion(
        canonical_substitution,
        source,
        images,
        counts,
        lengths,
        encoded_letter_bytes,
    )
    return _expand_polynomial_substitution(canonical_substitution, source, images)


def concatenate_words(
    left: FreeAlgebraWord, right: FreeAlgebraWord
) -> FreeAlgebraWordPairResult:
    """Concatenate two words over the same ordered generator alphabet."""

    left_value, right_value = _admit_word_pair(left, right)
    output_length = left_value.length + right_value.length
    if output_length > MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH:
        _reject_resource(
            ("right", "letters"),
            "word_result_length",
            f"concatenated word exceeds the {MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH}-letter value bound",
        )
    product_word = FreeAlgebraWord(
        alphabet=left_value.alphabet,
        letters=left_value.letters + right_value.letters,
    )
    return FreeAlgebraWordPairResult(
        left=left_value,
        right=right_value,
        product=product_word,
        left_range=(0, left_value.length),
        right_range=(left_value.length, output_length),
        degree_addition=(left_value.length, right_value.length, output_length),
    )


def power_word(word: FreeAlgebraWord, exponent: int) -> FreeAlgebraWordPowerResult:
    """Return an admitted nonnegative concatenation power of a source word."""

    value = _admit_word(word, label="word")
    if (
        not isinstance(exponent, int)
        or isinstance(exponent, bool)
        or not 0 <= exponent <= MAX_FREE_WORD_POWER_EXPONENT
    ):
        _reject_resource(
            ("exponent",),
            "word_power_exponent",
            "word power exponent must be an integer from 0 through the admitted bound",
        )
    output_length = value.length * exponent
    if output_length > MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH:
        _reject_resource(
            ("exponent",),
            "word_power_output_length",
            f"word power exceeds the {MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH}-letter output bound",
        )
    output = FreeAlgebraWord(
        alphabet=value.alphabet,
        letters=value.letters * exponent,
    )
    return FreeAlgebraWordPowerResult(
        word=value,
        exponent=exponent,
        power=output,
        degree_multiplication=(value.length, exponent, output_length),
    )


def reverse_word(word: FreeAlgebraWord) -> FreeAlgebraWordReverseResult:
    """Reverse the ordered generator sequence of one free word."""

    value = _admit_word(word, label="word")
    return FreeAlgebraWordReverseResult(
        word=value,
        reverse=FreeAlgebraWord(alphabet=value.alphabet, letters=value.letters[::-1]),
    )


def compare_words(
    left: FreeAlgebraWord, right: FreeAlgebraWord
) -> FreeAlgebraWordCompareResult:
    """Compare words in the algebra's degree-lexicographic monomial order."""

    left_value, right_value = _admit_word_pair(left, right)
    degree_comparison = cast(
        Literal[-1, 0, 1],
        (left_value.length > right_value.length)
        - (left_value.length < right_value.length),
    )
    first_difference = next(
        (
            index
            for index, (left_letter, right_letter) in enumerate(
                zip(left_value.letters, right_value.letters, strict=False)
            )
            if left_letter != right_letter
        ),
        None,
    )
    if first_difference is None:
        generator_comparison: Literal[-1, 0, 1] = 0
        left_rank = None
        right_rank = None
    else:
        left_rank = left_value.alphabet.index(left_value.letters[first_difference])
        right_rank = left_value.alphabet.index(right_value.letters[first_difference])
        generator_comparison = cast(
            Literal[-1, 0, 1], (left_rank > right_rank) - (left_rank < right_rank)
        )
    comparison = degree_comparison or generator_comparison
    return FreeAlgebraWordCompareResult(
        left=left_value,
        right=right_value,
        comparison=comparison,
        degree_comparison=degree_comparison,
        generator_order_comparison=generator_comparison,
        first_differing_position=first_difference,
        left_generator_rank=left_rank,
        right_generator_rank=right_rank,
    )


def word_prefixes(word: FreeAlgebraWord) -> FreeAlgebraWordPrefixesResult:
    """Return every prefix with its complementary suffix for reconstruction."""

    value = _admit_word(word, label="word")
    split_count = value.length + 1
    split_letter_cells = value.length * split_count
    if (
        split_count > MAX_FREE_WORD_SPLITS
        or split_letter_cells > MAX_FREE_WORD_SPLIT_LETTER_CELLS
    ):
        _reject_resource(
            ("word", "letters"),
            "prefix_output",
            "prefix family exceeds its admitted row or letter-cell bound",
        )
    splits = tuple(
        FreeAlgebraWordPrefixSplit(
            prefix_letters=value.letters[:cut],
            completing_suffix_letters=value.letters[cut:],
        )
        for cut in range(value.length + 1)
    )
    return FreeAlgebraWordPrefixesResult(word=value, splits=splits)


def word_suffixes(word: FreeAlgebraWord) -> FreeAlgebraWordSuffixesResult:
    """Return every suffix with its complementary prefix for reconstruction."""

    value = _admit_word(word, label="word")
    split_count = value.length + 1
    split_letter_cells = value.length * split_count
    if (
        split_count > MAX_FREE_WORD_SPLITS
        or split_letter_cells > MAX_FREE_WORD_SPLIT_LETTER_CELLS
    ):
        _reject_resource(
            ("word", "letters"),
            "suffix_output",
            "suffix family exceeds its admitted row or letter-cell bound",
        )
    splits = tuple(
        FreeAlgebraWordSuffixSplit(
            completing_prefix_letters=value.letters[:cut],
            suffix_letters=value.letters[cut:],
        )
        for cut in range(value.length + 1)
    )
    return FreeAlgebraWordSuffixesResult(word=value, splits=splits)


def word_factors(word: FreeAlgebraWord) -> FreeAlgebraWordFactorsResult:
    """Return distinct factors and all positions where each occurs."""

    value = _admit_word(word, label="word")
    count = (value.length + 1) * (value.length + 2) // 2
    if count > MAX_FREE_WORD_FACTOR_OCCURRENCES:
        _reject_resource(
            ("word", "letters"),
            "factor_count",
            "factor occurrences exceed the admitted quadratic result bound",
        )
    distinct_upper_bound = value.length * (value.length + 1) // 2 + 1
    if distinct_upper_bound > MAX_FREE_WORD_FACTOR_DISTINCT:
        _reject_resource(
            ("word", "letters"),
            "factor_distinct_count",
            "distinct factors exceed the admitted output bound",
        )
    occurrence_letter_cells = (
        value.length * (value.length + 1) * (value.length + 2) // 6
    )
    if occurrence_letter_cells > MAX_FREE_WORD_FACTOR_LETTER_CELLS:
        _reject_resource(
            ("word", "letters"),
            "factor_letter_cells",
            "factor content exceeds its admitted output-cell bound",
        )
    positions_by_factor: dict[tuple[str, ...], list[int]] = {}
    for start in range(value.length + 1):
        for end in range(start, value.length + 1):
            factor = value.letters[start:end]
            positions_by_factor.setdefault(factor, []).append(start)
    assert len(positions_by_factor) <= distinct_upper_bound
    factors = tuple(
        FreeAlgebraWordFactorOccurrences(letters=letters, positions=tuple(positions))
        for letters, positions in positions_by_factor.items()
    )
    return FreeAlgebraWordFactorsResult(word=value, factors=factors)


def word_overlaps(
    left: FreeAlgebraWord, right: FreeAlgebraWord
) -> FreeAlgebraWordOverlapResult:
    """Return every nontrivial compatible overlap or proper inclusion context."""

    left_value, right_value = _admit_word_pair(left, right)
    left_length, right_length = left_value.length, right_value.length
    candidate_count = (
        left_length + right_length - 1 if left_length and right_length else 0
    )
    if candidate_count > MAX_FREE_WORD_OVERLAP_ALIGNMENTS:
        _reject_resource(
            ("left", "letters"),
            "overlap_alignment_count",
            "word-pair overlap alignments exceed the admitted bound",
        )
    max_common_length = max(0, left_length + right_length - 1)
    alignment_work = candidate_count * (left_length + right_length)
    if alignment_work > MAX_FREE_WORD_OVERLAP_WORK:
        _reject_resource(
            ("left", "letters"),
            "overlap_comparison_work",
            "overlap compatibility checks exceed the admitted work bound",
        )
    witness_letter_cells = (
        candidate_count * (3 * max_common_length + len(left_value.alphabet))
        + left_length
        + right_length
        + 2 * len(left_value.alphabet)
    )
    if witness_letter_cells > MAX_FREE_WORD_OVERLAP_LETTER_CELLS:
        _reject_resource(
            ("left", "letters"),
            "overlap_letter_cells",
            "overlap witnesses exceed the admitted aggregate output-cell bound",
        )
    witnesses: list[FreeAlgebraWordOverlapWitness] = []
    for right_start in range(-right_length + 1, left_length) if candidate_count else ():
        left_start = 0
        common_start = min(left_start, right_start)
        common_end = max(left_length, right_start + right_length)
        common_length = common_end - common_start
        left_offset = left_start - common_start
        right_offset = right_start - common_start
        common: list[str | None] = [None] * common_length
        compatible = True
        for offset, letter in enumerate(left_value.letters):
            common[left_offset + offset] = letter
        for offset, letter in enumerate(right_value.letters):
            index = right_offset + offset
            if common[index] is not None and common[index] != letter:
                compatible = False
                break
            common[index] = letter
        if not compatible or any(letter is None for letter in common):
            continue
        if left_offset == right_offset and left_length == right_length:
            continue
        common_letters = tuple(letter for letter in common if letter is not None)
        left_end = left_offset + left_length
        right_end = right_offset + right_length
        inclusion = (left_offset <= right_offset and right_end <= left_end) or (
            right_offset <= left_offset and left_end <= right_end
        )
        witnesses.append(
            FreeAlgebraWordOverlapWitness(
                kind="INCLUSION" if inclusion else "OVERLAP",
                common_word=FreeAlgebraWord(
                    alphabet=left_value.alphabet, letters=common_letters
                ),
                left_offset=left_offset,
                right_offset=right_offset,
                left_prefix=common_letters[:left_offset],
                left_suffix=common_letters[left_end:],
                right_prefix=common_letters[:right_offset],
                right_suffix=common_letters[right_end:],
            )
        )
    return FreeAlgebraWordOverlapResult(
        left=left_value, right=right_value, witnesses=tuple(witnesses)
    )


def substitute_word(
    substitution: FreeAlgebraWordSubstitution,
    word: FreeAlgebraWord,
) -> FreeAlgebraWordSubstitutionResult:
    """Apply a specified free-monoid homomorphism to one source word."""

    try:
        substitution_value = FreeAlgebraWordSubstitution.model_validate(
            substitution.model_dump()
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("substitution",),
            code="free_algebra.substitution_shape",
            message="the word substitution is not canonical",
        ) from exc
    word_value = _admit_word(word, label="word")
    if word_value.alphabet != substitution_value.source_alphabet:
        raise OperationDomainValidationError(
            location=("word", "alphabet"),
            code="free_algebra.substitution_source_alphabet",
            message="word must use the substitution source alphabet",
        )
    images = substitution_value.images
    image_by_generator = dict(
        zip(substitution_value.source_alphabet, images, strict=True)
    )
    output_length = sum(
        image_by_generator[letter].length for letter in word_value.letters
    )
    if output_length > MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH:
        _reject_resource(
            ("word", "letters"),
            "substitution_output_length",
            f"substitution output exceeds the {MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH}-letter value bound",
        )
    output_letters = tuple(
        target_letter
        for letter in word_value.letters
        for target_letter in image_by_generator[letter].letters
    )
    image = FreeAlgebraWord(
        alphabet=substitution_value.target_alphabet, letters=output_letters
    )
    occurrence_images: list[FreeWordImageInterval] = []
    target_offset = 0
    for source_index, source_letter in enumerate(word_value.letters):
        image_length = image_by_generator[source_letter].length
        occurrence_images.append(
            FreeWordImageInterval(
                source_index=source_index,
                target_start=target_offset,
                target_end=target_offset + image_length,
            )
        )
        target_offset += image_length
    return FreeAlgebraWordSubstitutionResult(
        substitution=substitution_value,
        word=word_value,
        image=image,
        occurrence_images=tuple(occurrence_images),
    )


def _map(value: FreeAlgebraPolynomial) -> dict[tuple[str, ...], Fraction]:
    return {term.word: term.coefficient.as_fraction() for term in value.terms}


def _encode(
    alphabet: tuple[str, ...], values: Mapping[tuple[str, ...], Fraction]
) -> FreeAlgebraPolynomial:
    terms = tuple(
        FreeAlgebraTerm(
            coefficient=CanonicalRational.from_fraction(coefficient), word=word
        )
        for word, coefficient in sorted(
            ((word, value) for word, value in values.items() if value),
            key=lambda item: canonical_word_key(alphabet, item[0]),
            reverse=True,
        )
    )
    return FreeAlgebraPolynomial(alphabet=alphabet, terms=terms)


def _contexts(alphabet: tuple[str, ...], length: int) -> tuple[tuple[str, ...], ...]:
    return tuple(product(alphabet, repeat=length)) if length else ((),)


def _ideal_prefix(
    ideal: FreeAlgebraIdeal, degree: int
) -> tuple[FreeAlgebraPolynomial, ...]:
    values: list[FreeAlgebraPolynomial] = []
    for generator in ideal.generators:
        generator_degree = max((len(term.word) for term in generator.terms), default=0)
        if generator_degree > degree:
            continue
        context_length = degree - generator_degree
        if ideal.side == "two-sided":
            context_pairs = tuple(
                (left_context, right_context)
                for left_length in range(context_length + 1)
                for left_context in _contexts(ideal.alphabet, left_length)
                for right_context in _contexts(
                    ideal.alphabet, context_length - left_length
                )
            )
        elif ideal.side == "left":
            context_pairs = tuple(
                (context, ()) for context in _contexts(ideal.alphabet, context_length)
            )
        else:
            context_pairs = tuple(
                ((), context) for context in _contexts(ideal.alphabet, context_length)
            )
        for left_context, right_context in context_pairs:
            left = FreeAlgebraPolynomial(
                alphabet=ideal.alphabet,
                terms=(
                    FreeAlgebraTerm(
                        coefficient=CanonicalRational.from_fraction(Fraction(1)),
                        word=left_context,
                    ),
                ),
            )
            right = FreeAlgebraPolynomial(
                alphabet=ideal.alphabet,
                terms=(
                    FreeAlgebraTerm(
                        coefficient=CanonicalRational.from_fraction(Fraction(1)),
                        word=right_context,
                    ),
                ),
            )
            if ideal.side == "left":
                values.append(multiply(left, generator).product)
            elif ideal.side == "right":
                values.append(multiply(generator, right).product)
            else:
                values.append(
                    multiply(multiply(left, generator).product, right).product
                )
    return tuple(values)


def _admit_ideal(ideal: FreeAlgebraIdeal) -> FreeAlgebraIdeal:
    """Revalidate typed ideals before selecting a side or expanding them."""
    try:
        return FreeAlgebraIdeal.model_validate(ideal.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("ideal",),
            code="free_algebra.ideal_shape",
            message="the ideal presentation is not canonical",
        ) from exc


def _as_ideal(ideal: FreeAlgebraIdeal | Mapping[str, Any]) -> FreeAlgebraIdeal:
    try:
        parsed = (
            ideal
            if isinstance(ideal, FreeAlgebraIdeal)
            else FreeAlgebraIdeal.model_validate(ideal)
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("ideal",),
            code="free_algebra.ideal_shape",
            message="the ideal presentation is not canonical",
        ) from exc
    return _admit_ideal(parsed)


def ideal_generated_prefix(
    ideal: FreeAlgebraIdeal | Mapping[str, Any], degree: int
) -> FreeAlgebraIdealPrefixResult:
    value = _as_ideal(ideal)
    if (
        not isinstance(degree, int)
        or isinstance(degree, bool)
        or not 0 <= degree <= MAX_FREE_ALGEBRA_WORD_LENGTH
    ):
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="free_algebra.ideal_degree_bound",
            message="ideal prefix degree exceeds the admitted envelope",
        )
    if len(value.generators) > MAX_FREE_ALGEBRA_OPERAND_TERMS or any(
        len(generator.terms) > MAX_FREE_ALGEBRA_OPERAND_TERMS
        or any(
            len(term.word) > MAX_FREE_ALGEBRA_WORD_LENGTH for term in generator.terms
        )
        for generator in value.generators
    ):
        raise OperationResourceAdmissionError(
            location=("ideal",),
            code="free_algebra.ideal_generator_budget",
            message="ideal generator expansion exceeds the admitted envelope",
        )
    predicted_basis = 0
    predicted_terms = 0
    predicted_bytes = 0
    for generator in value.generators:
        generator_degree = max((len(term.word) for term in generator.terms), default=0)
        if generator_degree > degree:
            continue
        remaining = degree - generator_degree
        contexts = (
            len(value.alphabet) ** remaining
            if value.side != "two-sided"
            else (remaining + 1) * len(value.alphabet) ** remaining
        )
        generator_terms = len(generator.terms)
        predicted_basis += contexts
        predicted_terms += contexts * generator_terms
        # This is intentionally source-derived and conservative: every term
        # carries a coefficient and a word, plus the enclosing basis record.
        term_bytes = (
            2 * MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS
            + degree * (MAX_FREE_ALGEBRA_LETTER_LENGTH + 8)
            + 128
        )
        predicted_bytes += contexts * (generator_terms * term_bytes + 128)
        if any(
            canonical_rational_component_digits(term.coefficient)
            > MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS
            for term in generator.terms
        ):
            _reject_resource(
                ("ideal", "generators"),
                "ideal_prefix_coefficient_growth",
                "ideal prefix coefficients exceed the aggregate carrier budget",
            )
    if predicted_basis > MAX_FREE_ALGEBRA_IDEAL_PREFIX_BASIS:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="free_algebra.ideal_prefix_basis",
            message="ideal prefix basis cardinality exceeds the admitted envelope",
        )
    if predicted_terms > MAX_FREE_ALGEBRA_IDEAL_PREFIX_TOTAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="free_algebra.ideal_prefix_terms",
            message="ideal prefix aggregate terms exceed the admitted envelope",
        )
    if predicted_bytes > MAX_FREE_ALGEBRA_IDEAL_PREFIX_SERIALIZED_BYTES:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="free_algebra.ideal_prefix_serialized_size",
            message="ideal prefix serialized aggregate exceeds the admitted envelope",
        )
    basis = _ideal_prefix(value, degree)
    return FreeAlgebraIdealPrefixResult.model_construct(
        ideal=value, degree=degree, basis=basis
    )


def _component_generator_data(
    value: FreeAlgebraIdeal, degree: int
) -> tuple[list[tuple[FreeAlgebraPolynomial, int, int]], int, int]:
    generator_data: list[tuple[FreeAlgebraPolynomial, int, int]] = []
    context_count = 0
    max_integer_entry_digits = 1
    for index, generator in enumerate(value.generators):
        if len(generator.terms) > MAX_FREE_ALGEBRA_OPERAND_TERMS:
            _reject_resource(
                ("ideal", "generators", index),
                "component_generator_terms",
                "ideal component generators exceed the bounded term envelope",
            )
        if any(
            len(term.word) > MAX_FREE_ALGEBRA_WORD_LENGTH for term in generator.terms
        ):
            _reject_resource(
                ("ideal", "generators", index),
                "component_generator_word_length",
                "ideal component generator words exceed the admitted length",
            )
        if not generator.terms:
            continue
        generator_degree = len(generator.terms[0].word)
        if any(len(term.word) != generator_degree for term in generator.terms):
            raise OperationDomainValidationError(
                location=("ideal", "generators", index),
                code="free_algebra.component_homogeneity",
                message="ideal component generators must be homogeneous",
            )
        if generator_degree > degree:
            continue
        common_denominator = lcm(
            *(term.coefficient.as_fraction().denominator for term in generator.terms)
        )
        for term in generator.terms:
            coefficient = term.coefficient.as_fraction()
            scaled_numerator = coefficient.numerator * (
                common_denominator // coefficient.denominator
            )
            max_integer_entry_digits = max(
                max_integer_entry_digits,
                len(str(abs(scaled_numerator))),
            )
        remaining = degree - generator_degree
        context_count += (remaining + 1) * len(value.alphabet) ** remaining
        generator_data.append((generator, remaining, common_denominator))
    return generator_data, context_count, max_integer_entry_digits


def _admit_component_resources(
    value: FreeAlgebraIdeal,
    degree: int,
    context_count: int,
    max_integer_entry_digits: int,
) -> int:
    alphabet = value.alphabet
    ambient_dimension: int = len(alphabet) ** degree
    if ambient_dimension > MAX_FREE_ALGEBRA_IDEAL_COMPONENT_WORDS:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="free_algebra.component_word_axis",
            message=(
                "the degree word space exceeds the "
                f"{MAX_FREE_ALGEBRA_IDEAL_COMPONENT_WORDS}-word component bound"
            ),
        )
    if context_count > MAX_FREE_ALGEBRA_IDEAL_COMPONENT_CONTEXTS:
        raise OperationResourceAdmissionError(
            location=("ideal", "generators"),
            code="free_algebra.component_contexts",
            message=(
                "two-sided context multiples exceed the "
                f"{MAX_FREE_ALGEBRA_IDEAL_COMPONENT_CONTEXTS}-row component bound"
            ),
        )
    if (
        context_count * ambient_dimension
        > MAX_FREE_ALGEBRA_IDEAL_COMPONENT_MATRIX_CELLS
    ):
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="free_algebra.component_matrix_cells",
            message="ideal component matrix exceeds its admitted cell bound",
        )
    rank_bound = min(context_count, ambient_dimension)
    minor_digits = (
        rank_bound * max_integer_entry_digits
        + (len(str(factorial(rank_bound) - 1)) if rank_bound > 1 else 0)
        if rank_bound
        else 1
    )
    if minor_digits > MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("ideal", "generators"),
            code="free_algebra.component_coefficient_growth",
            message=(
                "the exact rational row-space bound exceeds the "
                f"{MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS}-digit result envelope"
            ),
        )
    max_letter_json_bytes = max(
        (len(json.dumps(letter, ensure_ascii=True)) for letter in alphabet), default=0
    )
    term_bytes = (
        2 * MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS
        + degree * (max_letter_json_bytes + 8)
        + 128
    )
    basis_bytes = rank_bound * ambient_dimension * term_bytes + 256
    source_bytes = 256 + sum(
        len(json.dumps(letter, ensure_ascii=True)) for letter in alphabet
    )
    for generator in value.generators:
        source_bytes += 128
        for term in generator.terms:
            source_bytes += 256 + sum(
                len(json.dumps(letter, ensure_ascii=True)) + 2 for letter in term.word
            )
    result_bytes = basis_bytes + source_bytes
    if result_bytes > MAX_FREE_ALGEBRA_IDEAL_COMPONENT_SERIALIZED_BYTES:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="free_algebra.component_serialized_size",
            message="ideal component basis exceeds its admitted serialized-size bound",
        )
    return ambient_dimension


def _component_rows(
    alphabet: tuple[str, ...],
    degree: int,
    generator_data: list[tuple[FreeAlgebraPolynomial, int, int]],
    ambient_dimension: int,
) -> tuple[tuple[tuple[str, ...], ...], list[list[Fraction]]]:
    words = tuple(product(alphabet, repeat=degree))
    word_index = {word: index for index, word in enumerate(words)}
    rows: list[list[Fraction]] = []
    for generator, remaining, common_denominator in generator_data:
        generator_terms = tuple(
            (
                term.word,
                term.coefficient.as_fraction().numerator
                * (common_denominator // term.coefficient.as_fraction().denominator),
            )
            for term in generator.terms
        )
        for left_length in range(remaining + 1):
            right_length = remaining - left_length
            for left_context in _contexts(alphabet, left_length):
                for right_context in _contexts(alphabet, right_length):
                    row = [Fraction(0)] * ambient_dimension
                    for word, coefficient in generator_terms:
                        row[word_index[left_context + word + right_context]] += (
                            Fraction(coefficient)
                        )
                    rows.append(row)
    return words, rows


def _rref(rows: list[list[Fraction]], width: int) -> list[list[Fraction]]:
    pivot_row = 0
    for column in range(width):
        pivot = next(
            (index for index in range(pivot_row, len(rows)) if rows[index][column]),
            None,
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        scale = rows[pivot_row][column]
        rows[pivot_row] = [entry / scale for entry in rows[pivot_row]]
        for index, row in enumerate(rows):
            if index == pivot_row or not row[column]:
                continue
            scale = row[column]
            rows[index] = [
                entry - scale * pivot_entry
                for entry, pivot_entry in zip(row, rows[pivot_row], strict=True)
            ]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return rows[:pivot_row]


def ideal_degree_component(
    ideal: FreeAlgebraIdeal | Mapping[str, Any], degree: int
) -> FreeAlgebraIdealDegreeComponentResult:
    """Compute an exact canonical basis for one homogeneous two-sided ideal part.

    This deliberately uses the finite word-space definition directly.  It is
    independent of the Groebner-Shirshov completion operation and therefore
    provides a bounded exact route for small degree components.
    """

    value = _as_ideal(ideal)
    if value.side != "two-sided":
        raise OperationDomainValidationError(
            location=("ideal", "side"),
            code="free_algebra.component_requires_two_sided",
            message="ideal degree components require side='two-sided'",
        )
    if (
        not isinstance(degree, int)
        or isinstance(degree, bool)
        or not 0 <= degree <= MAX_FREE_ALGEBRA_WORD_LENGTH
    ):
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="free_algebra.component_degree_bound",
            message="ideal component degree exceeds the admitted envelope",
        )

    generator_data, context_count, max_integer_entry_digits = _component_generator_data(
        value, degree
    )
    ambient_dimension = _admit_component_resources(
        value, degree, context_count, max_integer_entry_digits
    )
    words, rows = _component_rows(
        value.alphabet, degree, generator_data, ambient_dimension
    )
    rows = _rref(rows, ambient_dimension)
    basis = tuple(
        _encode(
            value.alphabet,
            {word: row[index] for index, word in enumerate(words) if row[index]},
        )
        for row in rows
    )
    return FreeAlgebraIdealDegreeComponentResult.model_construct(
        ideal=value,
        degree=degree,
        component_basis=basis,
        ambient_dimension=ambient_dimension,
        ideal_dimension=len(basis),
    )


def _leading(value: FreeAlgebraPolynomial) -> tuple[tuple[str, ...], Fraction] | None:
    if not value.terms:
        return None
    term = value.terms[0]
    return term.word, term.coefficient.as_fraction()


_MAX_GS_COEFFICIENT_COMPONENT = 10**MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS - 1


def _component_product_fits(left: int, right: int) -> bool:
    left = abs(left)
    right = abs(right)
    return not left or not right or left <= _MAX_GS_COEFFICIENT_COMPONENT // right


def _require_product_fits(left: Fraction, right: Fraction) -> None:
    if not left or not right:
        return
    left_num = abs(left.numerator)
    right_num = abs(right.numerator)
    left_den = left.denominator
    right_den = right.denominator
    cancel_left = gcd(left_num, right_den)
    cancel_right = gcd(right_num, left_den)
    if not (
        _component_product_fits(left_num // cancel_left, right_num // cancel_right)
        and _component_product_fits(left_den // cancel_right, right_den // cancel_left)
    ):
        _reject_resource(
            ("ideal", "generators"),
            "gs_coefficient_growth_budget",
            "GS coefficient multiplication exceeds the admitted exact-digit envelope",
        )


def _require_difference_fits(left: Fraction, right: Fraction) -> None:
    if left == right:
        return
    common = gcd(left.denominator, right.denominator)
    left_multiplier = right.denominator // common
    right_multiplier = left.denominator // common
    if not (
        _component_product_fits(left.numerator, left_multiplier)
        and _component_product_fits(right.numerator, right_multiplier)
        and _component_product_fits(left.denominator, left_multiplier)
    ):
        _reject_resource(
            ("ideal", "generators"),
            "gs_coefficient_growth_budget",
            "GS coefficient subtraction exceeds the admitted exact-digit envelope",
        )
    left_component = left.numerator * left_multiplier
    right_component = right.numerator * right_multiplier
    if (left_component < 0) != (right_component < 0) and abs(
        left_component
    ) > _MAX_GS_COEFFICIENT_COMPONENT - abs(right_component):
        _reject_resource(
            ("ideal", "generators"),
            "gs_coefficient_growth_budget",
            "GS coefficient subtraction exceeds the admitted exact-digit envelope",
        )


def _admitted_quotient(numerator: Fraction, denominator: Fraction) -> Fraction:
    reciprocal = Fraction(denominator.denominator, denominator.numerator)
    _require_product_fits(numerator, reciprocal)
    return numerator * reciprocal


def _subtract(
    left: Mapping[tuple[str, ...], Fraction],
    right: Mapping[tuple[str, ...], Fraction],
    scale: Fraction = Fraction(1),
) -> dict[tuple[str, ...], Fraction]:
    result = dict(left)
    for word, coefficient in right.items():
        _require_product_fits(scale, coefficient)
        scaled = scale * coefficient
        existing = result.get(word, Fraction(0))
        _require_difference_fits(existing, scaled)
        result[word] = existing - scaled
        if not result[word]:
            del result[word]
    return result


def _multiply_monomial(
    value: FreeAlgebraPolynomial, prefix: tuple[str, ...], suffix: tuple[str, ...]
) -> FreeAlgebraPolynomial:
    return _encode(
        value.alphabet,
        {
            prefix + word + suffix: coefficient
            for word, coefficient in _map(value).items()
        },
    )


def _normal_form(
    value: FreeAlgebraPolynomial, basis: tuple[FreeAlgebraPolynomial, ...], degree: int
) -> FreeAlgebraPolynomial:
    current = _map(value)
    changed = True
    reduction_steps = 0
    while changed:
        changed = False
        for reducer in basis:
            leading = _leading(reducer)
            if leading is None:
                continue
            leading_word, leading_coefficient = leading
            for word in tuple(sorted(current, key=lambda item: (len(item), item))):
                if len(word) > degree:
                    continue
                for start in range(len(word) - len(leading_word) + 1):
                    if word[start : start + len(leading_word)] != leading_word:
                        continue
                    factor = _admitted_quotient(current[word], leading_coefficient)
                    replacement = _multiply_monomial(
                        reducer, word[:start], word[start + len(leading_word) :]
                    )
                    reduction_steps += 1
                    if reduction_steps > MAX_FREE_ALGEBRA_GS_REDUCTION_STEPS:
                        raise OperationResourceAdmissionError(
                            location=("degree",),
                            code="free_algebra.gs_reduction_budget",
                            message="GS normal-form reduction exceeds its admitted work envelope",
                        )
                    current = _subtract(current, _map(replacement), factor)
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break
    return _encode(value.alphabet, current)


def _compositions(
    left: FreeAlgebraPolynomial, right: FreeAlgebraPolynomial, degree: int
) -> tuple[FreeAlgebraPolynomial, ...]:
    left_leading = _leading(left)
    right_leading = _leading(right)
    if left_leading is None or right_leading is None:
        return ()
    lw, lc = left_leading
    rw, rc = right_leading
    values: list[FreeAlgebraPolynomial] = []
    # Proper overlaps of leading words, plus inclusion ambiguities.
    for overlap in range(1, min(len(lw), len(rw))):
        if lw[-overlap:] == rw[:overlap]:
            first = _multiply_monomial(left, (), rw[overlap:])
            second = _multiply_monomial(right, lw[:-overlap], ())
            scale = _admitted_quotient(lc, rc)
            candidate = _encode(
                left.alphabet, _subtract(_map(first), _map(second), scale)
            )
            if max((len(term.word) for term in candidate.terms), default=0) <= degree:
                values.append(candidate)
    if len(lw) >= len(rw):
        for start in range(len(lw) - len(rw) + 1):
            if lw[start : start + len(rw)] != rw:
                continue
            first = left
            second = _multiply_monomial(right, lw[:start], lw[start + len(rw) :])
            scale = _admitted_quotient(lc, rc)
            candidate = _encode(
                left.alphabet, _subtract(_map(first), _map(second), scale)
            )
            if max((len(term.word) for term in candidate.terms), default=0) <= degree:
                values.append(candidate)
    return tuple(values)


def _gs_prefix_generators(
    ideal: FreeAlgebraIdeal, degree: int
) -> list[FreeAlgebraPolynomial]:
    """Validate homogeneous GS inputs and select generators in this prefix."""

    basis: list[FreeAlgebraPolynomial] = []
    for index, generator in enumerate(ideal.generators):
        if len(generator.terms) > MAX_FREE_ALGEBRA_OPERAND_TERMS or any(
            len(term.word) > MAX_FREE_ALGEBRA_WORD_LENGTH for term in generator.terms
        ):
            raise OperationResourceAdmissionError(
                location=("ideal", "generators", index),
                code="free_algebra.gs_generator_budget",
                message="GS generator expansion exceeds the admitted envelope",
            )
        if not generator.terms:
            continue
        generator_degree = len(generator.terms[0].word)
        if any(len(term.word) != generator_degree for term in generator.terms):
            raise OperationDomainValidationError(
                location=("ideal", "generators", index),
                code="free_algebra.gs_requires_homogeneous_generators",
                message="degree-bounded GS completion requires homogeneous generators",
            )
        if generator_degree <= degree:
            basis.append(generator)
    return basis


def groebner_shirshov_through_degree(
    ideal: FreeAlgebraIdeal | Mapping[str, Any], degree: int
) -> GroebnerShirshovResult:
    value = _as_ideal(ideal)
    if value.side != "two-sided":
        raise OperationDomainValidationError(
            location=("ideal", "side"),
            code="free_algebra.gs_requires_two_sided",
            message="Groebner-Shirshov completion is defined here for two-sided ideals",
        )
    if (
        not isinstance(degree, int)
        or isinstance(degree, bool)
        or not 0 <= degree <= MAX_FREE_ALGEBRA_WORD_LENGTH
    ):
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="free_algebra.gs_degree_bound",
            message="GS degree exceeds the admitted envelope",
        )
    # In a homogeneous presentation, generators above the requested degree do
    # not contribute to this prefix. Retaining them in the returned basis would
    # confuse a through-degree result with an unbounded basis claim.
    basis = _gs_prefix_generators(value, degree)
    compositions: list[FreeAlgebraPolynomial] = []
    # Completion is complete only at a fixed point.  In particular, (f, g)
    # and (g, f) are distinct ordered ambiguities in a noncommutative algebra;
    # considering only i <= j silently loses the reverse overlap.  The pair
    # and composition bounds are operation work bounds, not a stopping rule.
    seen = set(basis)
    while True:
        current_basis = tuple(basis)
        pair_count = len(current_basis) * len(current_basis)
        if pair_count > MAX_FREE_ALGEBRA_GS_PAIR_CHECKS:
            raise OperationResourceAdmissionError(
                location=("degree",),
                code="free_algebra.gs_pair_budget",
                message="GS ordered-pair completion exceeds its admitted work envelope",
            )
        additions: list[FreeAlgebraPolynomial] = []
        for left in current_basis:
            for right in current_basis:
                for candidate in _compositions(left, right, degree):
                    remainder = _normal_form(candidate, current_basis, degree)
                    compositions.append(remainder)
                    if len(compositions) > MAX_FREE_ALGEBRA_GS_COMPOSITIONS:
                        raise OperationResourceAdmissionError(
                            location=("degree",),
                            code="free_algebra.gs_composition_budget",
                            message="GS completion exceeds its admitted composition work envelope",
                        )
                    if (
                        remainder.terms
                        and remainder not in seen
                        and remainder not in additions
                    ):
                        additions.append(remainder)
        if not additions:
            # Every ordered pair of the final basis has now reduced every
            # degree-bounded overlap/inclusion composition to zero.  Only at
            # this fixed point may the result claim COMPLETE_THROUGH_DEGREE.
            break
        if len(basis) + len(additions) > MAX_FREE_ALGEBRA_RESULT_TERMS:
            raise OperationResourceAdmissionError(
                location=("degree",),
                code="free_algebra.gs_output",
                message="GS completion exceeds the admitted basis envelope",
            )
        basis.extend(additions)
        seen.update(additions)
    return GroebnerShirshovResult.model_construct(
        ideal=value, degree=degree, basis=tuple(basis), compositions=tuple(compositions)
    )


def ideal_membership(
    ideal: FreeAlgebraIdeal | Mapping[str, Any],
    polynomial: FreeAlgebraPolynomial | Mapping[str, Any],
) -> FreeAlgebraIdealMembershipResult:
    """Decide bounded membership using a completed homogeneous GS basis.

    Finite-degree completion is a decision procedure here only when every
    generator is homogeneous: then overlaps and reductions preserve total
    degree, so completion through the target's largest degree suffices. A
    resource bound that prevents completion returns UNKNOWN and never a
    membership conclusion.
    """

    value = _as_ideal(ideal)
    try:
        parsed_candidate = (
            polynomial
            if isinstance(polynomial, FreeAlgebraPolynomial)
            else FreeAlgebraPolynomial.model_validate(polynomial)
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="free_algebra.polynomial_shape",
            message="the membership polynomial is not canonical",
        ) from exc
    candidate = _admit_polynomial(parsed_candidate, label="polynomial")
    if value.side != "two-sided":
        raise OperationDomainValidationError(
            location=("ideal", "side"),
            code="free_algebra.membership_requires_two_sided",
            message="bounded GS membership requires side='two-sided'",
        )
    if candidate.alphabet != value.alphabet:
        raise OperationDomainValidationError(
            location=("polynomial", "alphabet"),
            code="free_algebra.membership_alphabet_mismatch",
            message="the membership polynomial must use the ideal alphabet",
        )
    for index, generator in enumerate(value.generators):
        if generator.terms:
            degree = len(generator.terms[0].word)
            if any(len(term.word) != degree for term in generator.terms):
                raise OperationDomainValidationError(
                    location=("ideal", "generators", index),
                    code="free_algebra.membership_requires_homogeneous_generators",
                    message="bounded membership requires homogeneous ideal generators",
                )

    target_degree = max((len(term.word) for term in candidate.terms), default=0)
    try:
        completion = groebner_shirshov_through_degree(value, target_degree)
        remainder = _normal_form(candidate, completion.basis, target_degree)
    except OperationResourceAdmissionError:
        return FreeAlgebraIdealMembershipResult.model_construct(
            ideal=value,
            polynomial=candidate,
            status="UNKNOWN",
            normal_form=None,
            completion_degree=None,
        )
    return FreeAlgebraIdealMembershipResult.model_construct(
        ideal=value,
        polynomial=candidate,
        status="MEMBER" if not remainder.terms else "NOT_MEMBER",
        normal_form=remainder,
        completion_degree=target_degree,
    )


def _admit_quotient_profile(ideal: FreeAlgebraIdeal, degree: int) -> None:
    """Preflight word enumeration and conservative serialized output size."""

    alphabet_size = len(ideal.alphabet)
    candidate_count = sum(alphabet_size**length for length in range(degree + 1))
    if candidate_count > MAX_FREE_ALGEBRA_QUOTIENT_PROFILE_CANDIDATES:
        _reject_resource(
            ("degree",),
            "quotient_profile_search_budget",
            "quotient profile word search exceeds its admitted candidate budget",
        )
    source_bytes = len(
        json.dumps(
            ideal.model_dump(mode="json"), ensure_ascii=True, separators=(",", ":")
        )
    )
    max_letter_bytes = max(
        (len(json.dumps(letter, ensure_ascii=True)) for letter in ideal.alphabet),
        default=2,
    )
    all_words_bytes = sum(
        alphabet_size**length * (length * (max_letter_bytes + 1) + 32)
        for length in range(degree + 1)
    )
    leading_words_bytes = MAX_FREE_ALGEBRA_RESULT_TERMS * (
        degree * (max_letter_bytes + 1) + 32
    )
    output_bound = (
        source_bytes + all_words_bytes + leading_words_bytes + 512 * (degree + 1)
    )
    if output_bound > MAX_FREE_ALGEBRA_QUOTIENT_PROFILE_OUTPUT_BYTES:
        _reject_resource(
            ("degree",),
            "quotient_profile_output_budget",
            "quotient profile exceeds its conservative serialized-output bound",
        )


def quotient_normal_word_profile(
    ideal: FreeAlgebraIdeal | Mapping[str, Any], degree: int
) -> FreeAlgebraQuotientProfileResult:
    """Return irreducible-word bases and Hilbert values through degree D.

    Completion is computed from the source ideal within this request. The
    operation therefore does not trust a serialized caller claim of
    ``COMPLETE_THROUGH_DEGREE`` when deriving the quotient normal words.
    """

    value = _as_ideal(ideal)
    if value.side != "two-sided":
        raise OperationDomainValidationError(
            location=("ideal", "side"),
            code="free_algebra.quotient_profile_requires_two_sided",
            message="quotient normal-word profiles require a two-sided ideal",
        )
    if (
        not isinstance(degree, int)
        or isinstance(degree, bool)
        or not 0 <= degree <= MAX_FREE_ALGEBRA_WORD_LENGTH
    ):
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="free_algebra.quotient_profile_degree_bound",
            message="quotient profile degree exceeds the admitted envelope",
        )
    _admit_quotient_profile(value, degree)
    completion = groebner_shirshov_through_degree(value, degree)
    leading_words = tuple(
        sorted(
            {
                leading[0]
                for polynomial in completion.basis
                if (leading := _leading(polynomial)) is not None
            },
            key=lambda word: canonical_word_key(value.alphabet, word),
        )
    )
    components: list[FreeAlgebraQuotientDegreeComponent] = []
    hilbert_function: list[int] = []
    for current_degree in range(degree + 1):
        normal_words = tuple(
            word
            for word in product(value.alphabet, repeat=current_degree)
            if not any(
                word[start : start + len(leading)] == leading
                for leading in leading_words
                for start in range(len(word) - len(leading) + 1)
            )
        )
        components.append(
            FreeAlgebraQuotientDegreeComponent(
                degree=current_degree, normal_words=normal_words
            )
        )
        hilbert_function.append(len(normal_words))
    return FreeAlgebraQuotientProfileResult.model_construct(
        ideal=value,
        degree=degree,
        leading_words=leading_words,
        components=tuple(components),
        hilbert_function=tuple(hilbert_function),
    )


__all__ = [
    "compare_words",
    "concatenate_words",
    "groebner_shirshov_through_degree",
    "ideal_degree_component",
    "ideal_generated_prefix",
    "ideal_membership",
    "multiply",
    "power_word",
    "quotient_normal_word_profile",
    "reverse_word",
    "substitute_word",
    "word_factors",
    "word_overlaps",
    "word_prefixes",
    "word_suffixes",
]
