"""Native exact free associative word, polynomial, and ideal operations.

Each operation performs its semantic admission before its exact kernel and
constructs the canonical result without replaying the computed mathematics.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from itertools import product
from math import ceil, factorial, gcd, lcm, log2
from typing import Any, Literal, cast

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian._execution import request_checkpoint
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras._kernel import add_sparse, multiply_sparse
from jacobian.math.free_algebras._models import (
    MAX_FREE_ALGEBRA_ADDITION_OUTPUT_CELLS,
    MAX_FREE_ALGEBRA_ADDITION_TERMS,
    MAX_FREE_ALGEBRA_ANTIAUTOMORPHISM_OUTPUT_CELLS,
    MAX_FREE_ALGEBRA_ANTIAUTOMORPHISM_WORK,
    MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS,
    MAX_FREE_ALGEBRA_FACTOR_DFA_OUTPUT_CELLS,
    MAX_FREE_ALGEBRA_FACTOR_DFA_PREFIX_CANDIDATES,
    MAX_FREE_ALGEBRA_FACTOR_DFA_STATES,
    MAX_FREE_ALGEBRA_FACTOR_DFA_WORK,
    MAX_FREE_ALGEBRA_FORBIDDEN_WORD_LETTERS,
    MAX_FREE_ALGEBRA_FORBIDDEN_WORDS,
    MAX_FREE_ALGEBRA_GENERATORS,
    MAX_FREE_ALGEBRA_GS_COMPOSITIONS,
    MAX_FREE_ALGEBRA_GS_PAIR_CHECKS,
    MAX_FREE_ALGEBRA_GS_REDUCTION_STEPS,
    MAX_FREE_ALGEBRA_IDEAL_COMPONENT_CELLS,
    MAX_FREE_ALGEBRA_IDEAL_COMPONENT_CONTEXTS,
    MAX_FREE_ALGEBRA_IDEAL_COMPONENT_MATRIX_CELLS,
    MAX_FREE_ALGEBRA_IDEAL_COMPONENT_WORDS,
    MAX_FREE_ALGEBRA_IDEAL_PREFIX_BASIS,
    MAX_FREE_ALGEBRA_IDEAL_PREFIX_CELLS,
    MAX_FREE_ALGEBRA_IDEAL_PREFIX_TOTAL_TERMS,
    MAX_FREE_ALGEBRA_LETTER_LENGTH,
    MAX_FREE_ALGEBRA_OPERAND_TERMS,
    MAX_FREE_ALGEBRA_POLYNOMIAL_POWER_EXPONENT,
    MAX_FREE_ALGEBRA_QUOTIENT_PROFILE_CANDIDATES,
    MAX_FREE_ALGEBRA_QUOTIENT_PROFILE_OUTPUT_CELLS,
    MAX_FREE_ALGEBRA_RESULT_TERMS,
    MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH,
    MAX_FREE_ALGEBRA_SUBSTITUTION_EXPANSIONS,
    MAX_FREE_ALGEBRA_SUBSTITUTION_OUTPUT_CELLS,
    MAX_FREE_ALGEBRA_SUBSTITUTION_WORK,
    MAX_FREE_ALGEBRA_TERM_PAIRS,
    MAX_FREE_ALGEBRA_TRUNCATED_QUOTIENT_OUTPUT_CELLS,
    MAX_FREE_ALGEBRA_TRUNCATED_QUOTIENT_TABLE_TERMS,
    MAX_FREE_ALGEBRA_TRUNCATED_QUOTIENT_WORK,
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
    TruncatedFreeAlgebraQuotient,
    canonical_word_key,
)
from jacobian.math.logic.finite_alphabet import FiniteAlphabet
from jacobian.math.logic.languages.regular.values import DFA, DFATransition


def _reject_resource(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=location,
        code=f"free_algebra.{code}",
        message=message,
    )


def _admit_word(value: FreeAlgebraWord, *, label: str) -> FreeAlgebraWord:
    """Reauthenticate one word within the 64-letter canonical value bound."""

    try:
        return FreeAlgebraWord.model_validate(value.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=(label,),
            code="free_algebra.word_shape",
            message="the free-algebra word is not canonical",
        ) from exc


def _admit_source_word(value: FreeAlgebraWord, *, label: str) -> FreeAlgebraWord:
    """Admit one word for a growing operation's 32-letter source bound."""

    admitted = _admit_word(value, label=label)
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
    left_value = _admit_source_word(left, label="left")
    right_value = _admit_source_word(right, label="right")
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
    left: FreeAlgebraPolynomial,
    right: FreeAlgebraPolynomial,
    *,
    operand_term_limit: int = MAX_FREE_ALGEBRA_OPERAND_TERMS,
    operand_word_length_limit: int = MAX_FREE_ALGEBRA_WORD_LENGTH,
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
        if len(polynomial.terms) > operand_term_limit:
            _reject_resource(
                (side, "terms"),
                "operand_term_budget",
                f"{side} operand exceeds the "
                f"{operand_term_limit}-term multiplication budget",
            )
        for index, term in enumerate(polynomial.terms):
            if len(term.word) > operand_word_length_limit:
                _reject_resource(
                    (side, "terms", index, "word"),
                    "operand_word_length_budget",
                    f"{side} operand word exceeds the "
                    f"{operand_word_length_limit}-letter multiplication "
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

    left_num = len(format_canonical_integer(abs(left.num)))
    right_num = len(format_canonical_integer(abs(right.num)))
    left_den = len(format_canonical_integer(left.den))
    right_den = len(format_canonical_integer(right.den))
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
        if len(polynomial.terms) > MAX_FREE_ALGEBRA_ADDITION_TERMS:
            _reject_resource(
                (side, "terms"),
                "addition_operand_term_budget",
                f"{side} operand exceeds the "
                f"{MAX_FREE_ALGEBRA_ADDITION_TERMS}-term canonical support bound",
            )
        for index, term in enumerate(polynomial.terms):
            if len(term.word) > MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH:
                _reject_resource(
                    (side, "terms", index, "word"),
                    "addition_operand_word_length_budget",
                    "addition operand words are limited to "
                    f"{MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH} letters",
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

    alphabet_scalars = sum(len(letter) for letter in left.alphabet)
    maximum_word_scalars = max(
        (
            sum(len(letter) for letter in term.word)
            for term in (*left.terms, *right.terms)
        ),
        default=0,
    )
    # The fixed term allowance covers the term record, coefficient sign and
    # separator, and word delimiters. Each output word is drawn unchanged
    # from an input, so the bound counts allocated scalar and digit cells.
    predicted_output_cells = (
        alphabet_scalars
        + 64
        + result_term_bound
        * (64 + maximum_word_scalars + 2 * (predicted_coefficient_digits + 1))
    )
    if predicted_output_cells > MAX_FREE_ALGEBRA_ADDITION_OUTPUT_CELLS:
        _reject_resource(
            ("left", "right"),
            "addition_output_cells_budget",
            "predicted polynomial sum exceeds the "
            f"{MAX_FREE_ALGEBRA_ADDITION_OUTPUT_CELLS}-cell allocation bound",
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


def _admit_anti_label(label: object, location: tuple[str | int, ...]) -> int:
    if type(label) is not str or not label:
        raise OperationDomainValidationError(
            location=location,
            code="free_algebra.polynomial_shape",
            message="the free-algebra polynomial is not canonical",
        )
    if len(label) > MAX_FREE_ALGEBRA_LETTER_LENGTH:
        _reject_resource(
            location,
            "antiautomorphism_work_budget",
            "polynomial reversal input is outside the admitted work envelope",
        )
    try:
        label.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise OperationDomainValidationError(
            location=location,
            code="free_algebra.polynomial_shape",
            message="the free-algebra polynomial is not canonical",
        ) from exc
    return len(label)


def _admit_anti_coefficient_digits(
    term: FreeAlgebraTerm, polynomial: FreeAlgebraPolynomial
) -> int:
    coefficient = getattr(term, "coefficient", None)
    if type(coefficient) is not CanonicalRational:
        _admit_polynomial(polynomial, label="polynomial")
        raise OperationDomainValidationError(
            location=("polynomial", "terms"),
            code="free_algebra.polynomial_shape",
            message="the free-algebra polynomial is not canonical",
        )
    numerator = getattr(coefficient, "num", None)
    denominator = getattr(coefficient, "den", None)
    if type(numerator) is not int or type(denominator) is not int:
        _admit_polynomial(polynomial, label="polynomial")
        raise OperationDomainValidationError(
            location=("polynomial", "terms"),
            code="free_algebra.polynomial_shape",
            message="the free-algebra polynomial is not canonical",
        )
    component_limit = 10**MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS
    if (
        abs(numerator).bit_length() > component_limit.bit_length()
        or denominator.bit_length() > component_limit.bit_length()
        or abs(numerator) >= component_limit
        or denominator >= component_limit
    ):
        _reject_resource(
            ("polynomial", "terms"),
            "antiautomorphism_work_budget",
            "polynomial coefficient exceeds the admitted digit bound",
        )
    return max(len(str(abs(numerator))), len(str(denominator)))


def _antiautomorphism_admission_work(value: FreeAlgebraPolynomial) -> int:
    """Bound label admission, canonical checks, reversal, and linear sorting.

    Only tuple shapes and bounded scalar lengths are inspected here. Oversized
    labels and coefficients are rejected before any rank-key construction.
    """

    if type(value) is not FreeAlgebraPolynomial:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="free_algebra.polynomial_shape",
            message="the free-algebra polynomial is not canonical",
        )
    alphabet = getattr(value, "alphabet", None)
    terms = getattr(value, "terms", None)
    if type(alphabet) is not tuple or type(terms) is not tuple:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="free_algebra.polynomial_shape",
            message="the free-algebra polynomial is not canonical",
        )
    if (
        len(alphabet) > MAX_FREE_ALGEBRA_GENERATORS
        or len(terms) > MAX_FREE_ALGEBRA_RESULT_TERMS
    ):
        _reject_resource(
            ("polynomial",),
            "antiautomorphism_work_budget",
            "polynomial reversal input is outside the admitted work envelope",
        )

    alphabet_characters = 0
    for letter in alphabet:
        alphabet_characters += _admit_anti_label(letter, ("polynomial", "alphabet"))

    term_count = len(terms)
    total_cells = 0
    word_characters = 0
    coefficient_digits = 0
    maximum_word_length = 0
    for term in terms:
        if type(term) is not FreeAlgebraTerm or not hasattr(term, "word"):
            _admit_polynomial(value, label="polynomial")
            raise OperationDomainValidationError(
                location=("polynomial", "terms"),
                code="free_algebra.polynomial_shape",
                message="the free-algebra polynomial is not canonical",
            )
        word = term.word
        if type(word) is not tuple:
            _admit_polynomial(value, label="polynomial")
            raise OperationDomainValidationError(
                location=("polynomial", "terms"),
                code="free_algebra.polynomial_shape",
                message="the free-algebra polynomial is not canonical",
            )
        word_length = len(word)
        if word_length > MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH:
            _reject_resource(
                ("polynomial", "terms"),
                "antiautomorphism_work_budget",
                "polynomial reversal input is outside the admitted work envelope",
            )
        for letter in word:
            word_characters += _admit_anti_label(letter, ("polynomial", "terms"))
        coefficient_digits += _admit_anti_coefficient_digits(term, value)
        total_cells += word_length
        maximum_word_length = max(maximum_word_length, word_length)

    # Dictionary admission and lookups are charged using the actual label
    # sizes. Canonical-key checks, reversal, and radix sorting each traverse
    # the bounded word cells linearly.
    label_work = 10 * (len(alphabet) + 1) * word_characters + 2 * alphabet_characters
    linear_work = 6 * total_cells + len(alphabet) * maximum_word_length
    coefficient_work = 4 * coefficient_digits + 2 * term_count
    return max(1, label_work + linear_work + coefficient_work)


def _reverse_canonical_terms(
    polynomial: FreeAlgebraPolynomial,
    letter_rank: dict[str, int],
) -> tuple[FreeAlgebraTerm, ...]:
    reversed_terms = tuple(
        FreeAlgebraTerm(coefficient=term.coefficient, word=tuple(reversed(term.word)))
        for term in polynomial.terms
    )
    groups: dict[int, list[FreeAlgebraTerm]] = {}
    for term in reversed_terms:
        groups.setdefault(len(term.word), []).append(term)
    ordered: list[FreeAlgebraTerm] = []
    for word_length in sorted(groups, reverse=True):
        group = groups[word_length]
        for position in range(word_length - 1, -1, -1):
            buckets: list[list[FreeAlgebraTerm]] = [[] for _ in polynomial.alphabet]
            for term in group:
                buckets[letter_rank[term.word[position]]].append(term)
            group = [term for bucket in reversed(buckets) for term in bucket]
        ordered.extend(group)
    return tuple(ordered)


def _admit_antiautomorphism_polynomial(
    polynomial: FreeAlgebraPolynomial,
) -> dict[str, int]:
    """Recheck this operation's relied-upon polynomial invariants linearly."""

    rank = {letter: index for index, letter in enumerate(polynomial.alphabet)}
    if len(rank) != len(polynomial.alphabet):
        raise OperationDomainValidationError(
            location=("polynomial", "alphabet"),
            code="free_algebra.polynomial_shape",
            message="the free-algebra polynomial is not canonical",
        )

    previous_key: tuple[int, tuple[int, ...]] | None = None
    for term in polynomial.terms:
        coefficient = term.coefficient
        if (
            coefficient.den <= 0
            or gcd(abs(coefficient.num), coefficient.den) != 1
            or coefficient.num == 0
        ):
            raise OperationDomainValidationError(
                location=("polynomial", "terms"),
                code="free_algebra.polynomial_shape",
                message="the free-algebra polynomial is not canonical",
            )
        ranks: list[int] = []
        for letter in term.word:
            letter_index = rank.get(letter)
            if letter_index is None:
                raise OperationDomainValidationError(
                    location=("polynomial", "terms"),
                    code="free_algebra.polynomial_shape",
                    message="the free-algebra polynomial is not canonical",
                )
            ranks.append(letter_index)
        key = (len(term.word), tuple(ranks))
        if previous_key is not None and previous_key <= key:
            raise OperationDomainValidationError(
                location=("polynomial", "terms"),
                code="free_algebra.polynomial_shape",
                message="the free-algebra polynomial is not canonical",
            )
        previous_key = key
    return rank


def reverse_polynomial_antiautomorphism(
    polynomial: FreeAlgebraPolynomial,
) -> FreeAlgebraPolynomial:
    """Extend word reversal linearly to the free algebra over QQ.

    This map fixes every scalar coefficient and reverses each monomial word.
    Consequently it is involutive and reverses multiplication order.
    """

    work = _antiautomorphism_admission_work(polynomial)
    if work > MAX_FREE_ALGEBRA_ANTIAUTOMORPHISM_WORK:
        _reject_resource(
            ("polynomial", "terms"),
            "antiautomorphism_work_budget",
            "polynomial admission, reversal, and canonical ordering exceed the "
            "work bound",
        )

    letter_rank = _admit_antiautomorphism_polynomial(polynomial)
    term_cells = sum(len(term.word) for term in polynomial.terms)
    maximum_cells = MAX_FREE_ALGEBRA_RESULT_TERMS * MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH
    if term_cells > maximum_cells:
        _reject_resource(
            ("polynomial", "terms"),
            "antiautomorphism_work_budget",
            "polynomial word support exceeds the admitted reversal work bound",
        )
    # Reversal preserves each word's cells and every coefficient is unchanged.
    # Bound the retained alphabet, cells, coefficient digits, and per-term
    # allowances before creating the reversed support. This allocation bound
    # counts canonical value cells, not serialized JSON bytes.
    alphabet_cells = sum(len(letter) for letter in polynomial.alphabet)
    predicted_output_cells = alphabet_cells + 64
    for term in polynomial.terms:
        coefficient_digits = canonical_rational_component_digits(term.coefficient)
        predicted_output_cells += 64 + len(term.word) + 2 * (coefficient_digits + 1)
        if predicted_output_cells > MAX_FREE_ALGEBRA_ANTIAUTOMORPHISM_OUTPUT_CELLS:
            _reject_resource(
                ("polynomial",),
                "antiautomorphism_output_budget",
                "predicted reversed-polynomial output exceeds the "
                f"{MAX_FREE_ALGEBRA_ANTIAUTOMORPHISM_OUTPUT_CELLS}-cell bound",
            )

    canonical_terms = _reverse_canonical_terms(polynomial, letter_rank)
    # The operation has admitted the input, preserved its coefficients and
    # alphabet, and constructed a sorted support with distinct reversed words.
    # Avoid replaying the polynomial model validator and its rank-key sort.
    return FreeAlgebraPolynomial.model_construct(
        alphabet=polynomial.alphabet, terms=canonical_terms
    )


def _admit_factor_avoidance_inputs(
    alphabet: tuple[str, ...],
    forbidden_factors: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Revalidate the canonical avoidance family without a transport request."""

    if type(alphabet) is not tuple or any(
        type(letter) is not str for letter in alphabet
    ):
        raise OperationDomainValidationError(
            location=("alphabet",),
            code="free_algebra.factor_avoidance_shape",
            message="the avoidance alphabet is not canonical",
        )
    if any(
        not letter or len(letter) > MAX_FREE_ALGEBRA_LETTER_LENGTH
        for letter in alphabet
    ):
        raise OperationDomainValidationError(
            location=("alphabet",),
            code="free_algebra.factor_avoidance_shape",
            message="the avoidance alphabet is not canonical",
        )
    if len(set(alphabet)) != len(alphabet):
        raise OperationDomainValidationError(
            location=("alphabet",),
            code="free_algebra.factor_avoidance_shape",
            message="the avoidance alphabet is not canonical",
        )
    # The request model caps the alphabet at MAX_FREE_ALGEBRA_GENERATORS and
    # validates each letter as a unicode scalar. Native callers bypass both, so
    # re-admit the same domain here: without this, 27..32 distinct generators
    # returned a DFA for an alphabet the published operation refuses, 33+ escaped
    # as a raw pydantic error while building transitions, and a surrogate label
    # bypassed the scalar check entirely.
    if len(alphabet) > MAX_FREE_ALGEBRA_GENERATORS:
        raise OperationDomainValidationError(
            location=("alphabet",),
            code="free_algebra.factor_avoidance_shape",
            message=(
                "the avoidance alphabet may contain at most "
                f"{MAX_FREE_ALGEBRA_GENERATORS} distinct generators"
            ),
        )
    for letter in alphabet:
        try:
            letter.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise OperationDomainValidationError(
                location=("alphabet",),
                code="free_algebra.factor_avoidance_shape",
                message=(
                    "the avoidance alphabet may contain only unicode scalar values"
                ),
            ) from exc
    if type(forbidden_factors) is not tuple or any(
        type(word) is not tuple or any(type(letter) is not str for letter in word)
        for word in forbidden_factors
    ):
        raise OperationDomainValidationError(
            location=("forbidden_factors",),
            code="free_algebra.factor_avoidance_shape",
            message="the forbidden-factor family is not canonical",
        )
    if len(forbidden_factors) > MAX_FREE_ALGEBRA_FORBIDDEN_WORDS:
        raise OperationDomainValidationError(
            location=("forbidden_factors",),
            code="free_algebra.factor_avoidance_shape",
            message="the forbidden-factor family is not canonical",
        )
    if any(
        len(word) > MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH for word in forbidden_factors
    ):
        raise OperationDomainValidationError(
            location=("forbidden_factors",),
            code="free_algebra.factor_avoidance_shape",
            message="the forbidden-factor family is not canonical",
        )
    if any(letter not in alphabet for word in forbidden_factors for letter in word):
        raise OperationDomainValidationError(
            location=("forbidden_factors",),
            code="free_algebra.factor_avoidance_shape",
            message="every forbidden factor must use the declared alphabet",
        )
    if sum(map(len, forbidden_factors)) > MAX_FREE_ALGEBRA_FORBIDDEN_WORD_LETTERS:
        raise OperationDomainValidationError(
            location=("forbidden_factors",),
            code="free_algebra.factor_avoidance_shape",
            message="the forbidden-factor family is not canonical",
        )
    return alphabet, forbidden_factors


def _contains_factor(word: tuple[int, ...], factor: tuple[int, ...]) -> bool:
    return len(factor) <= len(word) and any(
        word[start : start + len(factor)] == factor
        for start in range(len(word) - len(factor) + 1)
    )


def _ends_with(word: tuple[int, ...], suffix: tuple[int, ...]) -> bool:
    return len(suffix) <= len(word) and word[len(word) - len(suffix) :] == suffix


def _minimal_forbidden_factors(
    supplied: tuple[tuple[int, ...], ...], work_bound: int
) -> tuple[tuple[int, ...], ...]:
    ordered = sorted(set(supplied), key=lambda word: (len(word), word))
    max_pattern_length = max(map(len, ordered), default=0)
    if len(ordered) ** 2 * max_pattern_length**2 > work_bound:
        _reject_resource(
            ("forbidden_factors",),
            "factor_avoidance_work_bound",
            "forbidden-factor normalization exceeds the admitted work bound",
        )
    minimal: list[tuple[int, ...]] = []
    for index, word in enumerate(ordered):
        if index % 16 == 0:
            request_checkpoint("during factor-avoidance normalization")
        if not any(_contains_factor(word, factor) for factor in minimal):
            minimal.append(word)
    return tuple(sorted(minimal, key=lambda word: (len(word), word)))


def _factor_avoidance_states(
    patterns: tuple[tuple[int, ...], ...], state_bound: int
) -> tuple[tuple[tuple[int, ...], ...], int, int, tuple[int, ...]]:
    if patterns == ((),):
        return (), 0, 1, ()
    prefixes: set[tuple[int, ...]] = {()}
    for word in patterns:
        for length in range(len(word)):
            prefixes.add(word[:length])
    ordered_prefixes = tuple(sorted(prefixes, key=lambda word: (len(word), word)))
    dead_state = len(ordered_prefixes)
    state_count = dead_state + (1 if patterns else 0)
    if state_count > state_bound:
        _reject_resource(
            ("forbidden_factors",),
            "factor_avoidance_state_bound",
            f"factor-avoidance DFA exceeds the {state_bound}-state carrier bound",
        )
    return ordered_prefixes, dead_state, state_count, tuple(range(dead_state))


def _preflight_factor_prefix_states(
    patterns: tuple[tuple[int, ...], ...],
    state_bound: int,
    prefix_candidate_bound: int,
) -> int:
    """Admit a conservative prefix bound, then count trie nodes without expansion.

    The length sum bounds every distinct proper-prefix node. Because
    ``patterns`` is a factor antichain, it is also prefix-free; adjacent words
    in lexical order therefore give the exact number of shared proper-prefix
    nodes from their lengths and longest common prefixes.
    """

    if patterns == ((),):
        return 1
    prefix_upper_bound = 1 + sum(max(0, len(word) - 1) for word in patterns)
    state_upper_bound = prefix_upper_bound + (1 if patterns else 0)
    if state_upper_bound > prefix_candidate_bound:
        _reject_resource(
            ("forbidden_factors",),
            "factor_avoidance_prefix_bound",
            "factor-avoidance prefix candidates exceed the admitted intermediate bound",
        )

    prefix_count = 1
    previous: tuple[int, ...] | None = None
    for word in sorted(patterns):
        common = 0
        if previous is not None:
            limit = min(len(previous), len(word))
            while common < limit and previous[common] == word[common]:
                common += 1
        prefix_count += max(0, len(word) - 1 - common)
        previous = word
    predicted_states = prefix_count + (1 if patterns else 0)
    if predicted_states > state_bound:
        _reject_resource(
            ("forbidden_factors",),
            "factor_avoidance_state_bound",
            f"factor-avoidance DFA exceeds the {state_bound}-state carrier bound",
        )
    return predicted_states


def _factor_avoidance_target(
    prefix: tuple[int, ...],
    symbol: int,
    patterns: tuple[tuple[int, ...], ...],
    prefix_index: dict[tuple[int, ...], int],
    dead_state: int,
    max_prefix_length: int,
) -> int:
    candidate = (*prefix, symbol)
    if any(_ends_with(candidate, pattern) for pattern in patterns):
        return dead_state
    for length in range(min(len(candidate), max_prefix_length), -1, -1):
        suffix = candidate[-length:] if length else ()
        if suffix in prefix_index:
            return prefix_index[suffix]
    return 0


def _factor_avoidance_transitions(
    prefixes: tuple[tuple[int, ...], ...],
    patterns: tuple[tuple[int, ...], ...],
    alphabet_size: int,
    dead_state: int,
) -> tuple[DFATransition, ...]:
    if patterns == ((),):
        return tuple(
            DFATransition(source=0, symbol=symbol, target=0)
            for symbol in range(alphabet_size)
        )
    prefix_index = {word: index for index, word in enumerate(prefixes)}
    max_prefix_length = max(map(len, prefixes), default=0)
    transitions: list[DFATransition] = []
    for state, prefix in enumerate(prefixes):
        if state % 16 == 0:
            request_checkpoint("during factor-avoidance transition construction")
        for symbol in range(alphabet_size):
            transitions.append(
                DFATransition(
                    source=state,
                    symbol=symbol,
                    target=_factor_avoidance_target(
                        prefix,
                        symbol,
                        patterns,
                        prefix_index,
                        dead_state,
                        max_prefix_length,
                    ),
                )
            )
    if patterns:
        transitions.extend(
            DFATransition(source=dead_state, symbol=symbol, target=dead_state)
            for symbol in range(alphabet_size)
        )
    return tuple(transitions)


def factor_avoidance_dfa(
    alphabet: tuple[str, ...],
    forbidden_factors: tuple[tuple[str, ...], ...],
) -> DFA:
    """Build the total DFA for words avoiding a finite set of contiguous factors.

    States record the longest suffix of the scanned word that is a proper
    prefix of a forbidden factor. A sink records that a forbidden factor has
    occurred. Generator labels map to DFA symbols by their declared alphabet
    rank. No quotient or Gröbner-completion claim is made.
    """

    alphabet, forbidden_factors = _admit_factor_avoidance_inputs(
        alphabet, forbidden_factors
    )
    supplied = tuple(
        tuple(alphabet.index(letter) for letter in word) for word in forbidden_factors
    )
    supplied_letter_count = sum(map(len, supplied))
    if (
        len(supplied) > MAX_FREE_ALGEBRA_FORBIDDEN_WORDS
        or supplied_letter_count > MAX_FREE_ALGEBRA_FORBIDDEN_WORD_LETTERS
    ):
        _reject_resource(
            ("forbidden_factors",),
            "factor_avoidance_input_bound",
            "forbidden-factor input exceeds the admitted family or letter bound",
        )

    # Canonicalize the family to its unique minimal factor antichain. Removing
    # a word that contains another forbidden word preserves exactly the same
    # avoidance language.
    minimal_tuple = _minimal_forbidden_factors(
        supplied, MAX_FREE_ALGEBRA_FACTOR_DFA_WORK
    )
    canonical_source_patterns = tuple(
        sorted(set(supplied), key=lambda word: (len(word), word))
    )
    labels = tuple(alphabet)
    predicted_state_count = _preflight_factor_prefix_states(
        minimal_tuple,
        MAX_FREE_ALGEBRA_FACTOR_DFA_STATES,
        MAX_FREE_ALGEBRA_FACTOR_DFA_PREFIX_CANDIDATES,
    )
    state_prefixes, dead_state, state_count, accepting = _factor_avoidance_states(
        minimal_tuple, MAX_FREE_ALGEBRA_FACTOR_DFA_STATES
    )
    if state_count != predicted_state_count:
        raise RuntimeError("factor-avoidance prefix preflight disagrees with trie size")
    alphabet_size = len(labels)
    transition_count = state_count * alphabet_size
    if transition_count > 4096:
        _reject_resource(
            ("alphabet",),
            "factor_avoidance_transition_bound",
            "factor-avoidance DFA exceeds the regular-language transition bound",
        )
    max_prefix_length = max(map(len, state_prefixes), default=0)
    minimal_letter_count = sum(map(len, minimal_tuple))
    transition_work = transition_count * (
        minimal_letter_count + len(state_prefixes) * max_prefix_length + 1
    )
    total_work = (
        len(supplied) ** 2 * max((len(word) for word in supplied), default=0) ** 2
        + transition_work
    )
    if total_work > MAX_FREE_ALGEBRA_FACTOR_DFA_WORK:
        _reject_resource(
            ("forbidden_factors",),
            "factor_avoidance_work_bound",
            "factor-avoidance DFA construction exceeds the admitted work bound",
        )
    output_cells = (
        sum(map(len, labels))
        + sum(len(labels[rank]) for word in canonical_source_patterns for rank in word)
        + 24 * len(canonical_source_patterns)
        + transition_count * 48
        + state_count * 24
        + 256
    )
    if output_cells > MAX_FREE_ALGEBRA_FACTOR_DFA_OUTPUT_CELLS:
        _reject_resource(
            ("forbidden_factors",),
            "factor_avoidance_output_bound",
            "factor-avoidance DFA exceeds the admitted output allocation",
        )

    transitions = _factor_avoidance_transitions(
        state_prefixes,
        minimal_tuple,
        alphabet_size,
        dead_state,
    )
    initial_state = 0

    dfa = DFA(
        state_count=state_count,
        alphabet_size=alphabet_size,
        alphabet=FiniteAlphabet(symbols=labels) if labels else None,
        transitions=tuple(transitions),
        initial_state=initial_state,
        accepting_states=accepting,
    )
    return dfa


def _admit_substitution_images(
    substitution: FreeAlgebraPolynomialHomomorphism,
    *,
    location_prefix: tuple[str | int, ...] = ("substitution",),
) -> tuple[dict[str, tuple[FreeAlgebraTerm, ...]], dict[str, int], dict[str, int], int]:
    """Reauthenticate images and gather the bounded expansion metadata."""
    image_term_counts: dict[str, int] = {}
    image_word_lengths: dict[str, int] = {}
    image_terms: dict[str, tuple[FreeAlgebraTerm, ...]] = {}
    maximum_letter_scalars = max(
        (len(letter) for letter in substitution.target_alphabet),
        default=0,
    )
    for letter, image in zip(
        substitution.source_alphabet,
        substitution.images,
        strict=True,
    ):
        if len(image.terms) > MAX_FREE_ALGEBRA_OPERAND_TERMS:
            _reject_resource(
                (*location_prefix, "images", letter, "terms"),
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
                    (*location_prefix, "images", letter, "terms", index, "word"),
                    "substitution_image_word_budget",
                    "generator image words are limited to 32 letters",
                )
    return (
        image_terms,
        image_term_counts,
        image_word_lengths,
        maximum_letter_scalars,
    )


def _preflight_substitution_expansion(
    substitution: FreeAlgebraPolynomialHomomorphism,
    sources: tuple[FreeAlgebraPolynomial, ...],
    image_terms: dict[str, tuple[FreeAlgebraTerm, ...]],
    image_term_counts: dict[str, int],
    image_word_lengths: dict[str, int],
    maximum_letter_scalars: int,
    *,
    source_locations: tuple[tuple[str | int, ...], ...] | None = None,
) -> None:
    """Admit expansion, coefficient, work, and result envelopes up front."""

    expansion_count = 0
    maximum_source_word_length = 0
    maximum_output_word_length = 0
    source_expansion_counts = []
    source_coefficient_digit_bounds = []
    locations = source_locations or tuple(("polynomial", "terms") for _ in sources)
    for source_index, source in enumerate(sources):
        location = locations[source_index]
        source_expansion_count = 0
        source_maximum_contribution_digits = 0
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
                        (
                            0
                            if abs(image_term.coefficient.num) == 1
                            and image_term.coefficient.den == 1
                            else canonical_rational_component_digits(
                                image_term.coefficient
                            )
                        )
                        for image_term in image_terms[letter]
                    ),
                    default=0,
                )
            expansion_count += expansion
            source_expansion_count += expansion
            if expansion:
                maximum_output_word_length = max(
                    maximum_output_word_length, output_length
                )
            if expansion:
                source_maximum_contribution_digits = max(
                    source_maximum_contribution_digits, contribution_digits
                )
        source_coefficient_digits = (
            source_expansion_count * source_maximum_contribution_digits
            + (len(str(source_expansion_count)) if source_expansion_count > 1 else 0)
        )
        source_expansion_counts.append(source_expansion_count)
        source_coefficient_digit_bounds.append(source_coefficient_digits)
        if expansion_count > MAX_FREE_ALGEBRA_SUBSTITUTION_EXPANSIONS:
            _reject_resource(
                location,
                "substitution_expansion_budget",
                "polynomial substitution exceeds the admitted expansion count",
            )

    if maximum_output_word_length > MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH:
        _reject_resource(
            locations[0],
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
            locations[0],
            "substitution_work_budget",
            "polynomial substitution exceeds the admitted exact work bound",
        )
    if any(count > MAX_FREE_ALGEBRA_RESULT_TERMS for count in source_expansion_counts):
        _reject_resource(
            next(
                location
                for location, count in zip(
                    locations, source_expansion_counts, strict=True
                )
                if count > MAX_FREE_ALGEBRA_RESULT_TERMS
            ),
            "substitution_result_term_budget",
            "polynomial substitution can exceed the exact result term limit",
        )
    if any(
        digits > MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS
        for digits in source_coefficient_digit_bounds
    ):
        _reject_resource(
            next(
                location
                for location, digits in zip(
                    locations, source_coefficient_digit_bounds, strict=True
                )
                if digits > MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS
            ),
            "substitution_coefficient_growth",
            "predicted exact coefficient growth exceeds the 64-digit limit",
        )
    output_cell_bound = expansion_count * (
        128
        + maximum_output_word_length * (maximum_letter_scalars + 4)
        + 2 * (MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS + 1)
    )
    if output_cell_bound > MAX_FREE_ALGEBRA_SUBSTITUTION_OUTPUT_CELLS:
        _reject_resource(
            locations[0],
            "substitution_output_cells",
            "predicted canonical output exceeds the admitted "
            f"{MAX_FREE_ALGEBRA_SUBSTITUTION_OUTPUT_CELLS}-cell allocation bound",
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
    images, counts, lengths, letter_scalars = _admit_substitution_images(
        canonical_substitution
    )
    _preflight_substitution_expansion(
        canonical_substitution,
        (source,),
        images,
        counts,
        lengths,
        letter_scalars,
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

    value = _admit_source_word(word, label="word")
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

    left_value = _admit_word(left, label="left")
    right_value = _admit_word(right, label="right")
    if left_value.alphabet != right_value.alphabet:
        raise OperationDomainValidationError(
            location=("right", "alphabet"),
            code="free_algebra.word_alphabet_mismatch",
            message="both words must use the same ordered generator alphabet",
        )
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
    if len(positions_by_factor) > distinct_upper_bound:
        _reject_resource(
            ("word", "letters"),
            "factor_distinct_count",
            "distinct factors exceed the admitted output bound",
        )
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
    word_value = _admit_source_word(word, label="word")
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
    predicted_cells = 0
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
        term_cells = (
            2 * MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS
            + degree * (MAX_FREE_ALGEBRA_LETTER_LENGTH + 8)
            + 128
        )
        predicted_cells += contexts * (generator_terms * term_cells + 128)
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
    if predicted_cells > MAX_FREE_ALGEBRA_IDEAL_PREFIX_CELLS:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="free_algebra.ideal_prefix_cells",
            message="ideal prefix aggregate cells exceed the admitted envelope",
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
    max_letter_scalars = max((len(letter) for letter in alphabet), default=0)
    term_cells = (
        2 * MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS
        + degree * (max_letter_scalars + 8)
        + 128
    )
    basis_cells = rank_bound * ambient_dimension * term_cells + 256
    source_cells = 256 + sum(len(letter) for letter in alphabet)
    for generator in value.generators:
        source_cells += 128
        for term in generator.terms:
            source_cells += 256 + sum(len(letter) + 2 for letter in term.word)
    result_cells = basis_cells + source_cells
    if result_cells > MAX_FREE_ALGEBRA_IDEAL_COMPONENT_CELLS:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="free_algebra.component_cells",
            message="ideal component basis exceeds its admitted cell-allocation bound",
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
                    # Bound the potential union support before subtraction allocates it.
                    if (
                        len(current) - 1 + len(reducer.terms)
                        > MAX_FREE_ALGEBRA_RESULT_TERMS
                    ):
                        raise OperationResourceAdmissionError(
                            location=("degree",),
                            code="free_algebra.gs_normal_form_support_budget",
                            message="GS normal-form support exceeds its admitted carrier",
                        )
                    request_checkpoint("during GS normal-form reduction")
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
            if len(generator.terms) > MAX_FREE_ALGEBRA_OPERAND_TERMS or any(
                len(term.word) > MAX_FREE_ALGEBRA_WORD_LENGTH
                for term in generator.terms
            ):
                raise OperationResourceAdmissionError(
                    location=("ideal", "generators", index),
                    code="free_algebra.gs_generator_budget",
                    message="GS generator expansion exceeds the admitted envelope",
                )
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
    """Preflight word enumeration and the conservative output allocation."""

    alphabet_size = len(ideal.alphabet)
    candidate_count = sum(alphabet_size**length for length in range(degree + 1))
    if candidate_count > MAX_FREE_ALGEBRA_QUOTIENT_PROFILE_CANDIDATES:
        _reject_resource(
            ("degree",),
            "quotient_profile_search_budget",
            "quotient profile word search exceeds its admitted candidate budget",
        )
    # The result echoes the ideal carrier, so bound its stored scalar and
    # digit cells alongside the enumerated words and leading-word list.
    source_cells = sum(len(letter) for letter in ideal.alphabet)
    for generator in ideal.generators:
        source_cells += 128
        for term in generator.terms:
            source_cells += (
                128
                + sum(len(letter) + 2 for letter in term.word)
                + 2 * (MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS + 1)
            )
    max_letter_scalars = max((len(letter) for letter in ideal.alphabet), default=0)
    all_words_cells = sum(
        alphabet_size**length * (length * (max_letter_scalars + 1) + 32)
        for length in range(degree + 1)
    )
    # Leading words are distinct enumerated candidates of degree at most
    # ``degree``, so ``candidate_count`` (not the polynomial result-term limit)
    # bounds their cardinality.
    leading_words_cells = candidate_count * (degree * (max_letter_scalars + 1) + 32)
    output_cells = (
        source_cells + all_words_cells + leading_words_cells + 512 * (degree + 1)
    )
    if output_cells > MAX_FREE_ALGEBRA_QUOTIENT_PROFILE_OUTPUT_CELLS:
        _reject_resource(
            ("degree",),
            "quotient_profile_output_budget",
            "quotient profile exceeds its admitted allocation-cell bound",
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


def _normal_words_through_degree(
    ideal: FreeAlgebraIdeal,
    degree: int,
    leading_words: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    words = []
    visited = 0
    for word_degree in range(degree + 1):
        for word in product(ideal.alphabet, repeat=word_degree):
            visited += 1
            if visited % 4_096 == 0:
                request_checkpoint("during truncated quotient normal-word enumeration")
            if not any(
                word[start : start + len(leading)] == leading
                for leading in leading_words
                for start in range(len(word) - len(leading) + 1)
            ):
                words.append(word)
    return tuple(words)


def _is_reducible_word(
    word: tuple[str, ...], leading_words: tuple[tuple[str, ...], ...]
) -> bool:
    return any(
        word[start : start + len(leading)] == leading
        for leading in leading_words
        for start in range(len(word) - len(leading) + 1)
    )


def _reduction_ratio_component_digits(
    basis: tuple[FreeAlgebraPolynomial, ...],
) -> int:
    """Bound component digits of one reducer tail-to-leading coefficient ratio.

    A normal-form rewrite replaces the accumulated coefficient ``C`` by
    ``-C * tail / leading`` for every tail term of the reducer.  The quotient
    between the reducer's coefficients, not either coefficient's individual
    width, drives component growth, so bind the cross-product components of
    each leading term and its tail terms before any reduction executes.
    """

    digits = 1
    for polynomial in basis:
        leading = _leading(polynomial)
        if leading is None:
            continue
        _, leading_coefficient = leading
        for term in polynomial.terms[1:]:
            tail = term.coefficient.as_fraction()
            digits = max(
                digits,
                len(str(abs(tail.numerator * leading_coefficient.denominator))),
                len(str(abs(tail.denominator * leading_coefficient.numerator))),
            )
    return digits


def _admit_truncated_table(
    ideal: FreeAlgebraIdeal,
    completion: GroebnerShirshovResult,
    degree: int,
    basis: tuple[tuple[str, ...], ...],
    leading_words: tuple[tuple[str, ...], ...],
) -> frozenset[tuple[int, int]]:
    """Preflight table cells, reduction work, and complete serialized output."""
    dimension = len(basis)
    pair_count = dimension**2
    term_count_bound = dimension**3
    if term_count_bound > MAX_FREE_ALGEBRA_TRUNCATED_QUOTIENT_TABLE_TERMS:
        _reject_resource(
            ("degree",),
            "truncated_quotient_table_terms",
            f"multiplication output may contain {term_count_bound} terms, exceeding the "
            f"{MAX_FREE_ALGEBRA_TRUNCATED_QUOTIENT_TABLE_TERMS}-term bound",
        )

    scan_work = pair_count * (degree + 1 + len(leading_words) * (degree + 1) ** 2)
    if scan_work > MAX_FREE_ALGEBRA_TRUNCATED_QUOTIENT_WORK:
        _reject_resource(
            ("degree",),
            "truncated_quotient_scan_work",
            "normal-word boundary checks exceed the truncated quotient work envelope",
        )
    reducible_pairs = set()
    for left_index, left_word in enumerate(basis):
        for right_index, right_word in enumerate(basis):
            if len(left_word) + len(right_word) <= degree and _is_reducible_word(
                left_word + right_word, leading_words
            ):
                reducible_pairs.add((left_index, right_index))
            if (left_index * max(1, dimension) + right_index) % 4_096 == 0:
                request_checkpoint("during truncated quotient multiplication preflight")

    candidate_count = sum(
        len(ideal.alphabet) ** word_degree for word_degree in range(degree + 1)
    )
    basis_term_count = sum(len(polynomial.terms) for polynomial in completion.basis)
    sort_work = (
        candidate_count * max(1, ceil(log2(max(2, candidate_count)))) * (degree + 1)
    )
    reduction_work = candidate_count * (
        candidate_count * len(completion.basis) * (degree + 1)
        + sort_work
        + basis_term_count
    )
    # Each reduction may multiply coefficients from input relations. Bound the
    # largest mandatory product before any table construction begins.
    coefficient_digits = max(
        (
            canonical_rational_component_digits(term.coefficient)
            for p in completion.basis
            for term in p.terms
        ),
        default=1,
    )
    # A normal-form rewrite replaces the accumulated coefficient C by
    # -C * (tail / leading) for every reducer tail term.  The quotient between
    # the reducer coefficients, not their individual component widths, drives
    # growth: a leading coefficient 1/10^8 beside a 10^8 tail multiplies by
    # 10^16 per crossing.  Across an at-most-degree-long reduction chain, bound
    # every such ratio before constructing the multiplication table.
    ratio_digits = _reduction_ratio_component_digits(completion.basis)
    predicted_coefficient_digits = (degree + 1) * max(coefficient_digits, ratio_digits)
    if predicted_coefficient_digits > MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS:
        _reject_resource(
            ("degree",),
            "gs_coefficient_growth_budget",
            "truncated quotient reduction may exceed the admitted coefficient digit envelope",
        )
    total_work = scan_work + len(reducible_pairs) * reduction_work
    if total_work > MAX_FREE_ALGEBRA_TRUNCATED_QUOTIENT_WORK:
        _reject_resource(
            ("degree",),
            "truncated_quotient_reduction_work",
            f"bounded multiplication reduction needs at most {total_work} work units, "
            f"exceeding {MAX_FREE_ALGEBRA_TRUNCATED_QUOTIENT_WORK}",
        )

    # Admission counts the exact value the operation will return, in cells
    # of one word letter or one rational component. Estimating a serialized
    # transport size here would let a consumer's encoder choice, not the
    # mathematics, decide admission. Coefficient width is already bounded
    # above by the admitted digit envelope, so cells determine the value.
    def _polynomial_cells(polynomial: FreeAlgebraPolynomial) -> int:
        return sum(len(term.word) + 2 for term in polynomial.terms)

    ideal_cells = sum(_polynomial_cells(generator) for generator in ideal.generators)
    completion_cells = sum(
        _polynomial_cells(polynomial)
        for polynomial in (*completion.basis, *completion.compositions)
    )
    basis_cells = sum(len(word) for word in basis)
    # Each reducible pair stores one exact product, bounded by the admitted
    # per-pair term bound over words of at most ``degree`` letters.
    table_cells = pair_count * 2 * term_count_bound * (degree + 1)
    output_cells = (
        ideal_cells + completion_cells + basis_cells + table_cells + degree + 1
    )
    if output_cells > MAX_FREE_ALGEBRA_TRUNCATED_QUOTIENT_OUTPUT_CELLS:
        _reject_resource(
            ("degree",),
            "truncated_quotient_output_cells",
            f"truncated quotient allocates at most {output_cells} exact cells, "
            f"exceeding the {MAX_FREE_ALGEBRA_TRUNCATED_QUOTIENT_OUTPUT_CELLS}-cell "
            "allocation bound",
        )
    return frozenset(reducible_pairs)


def _truncated_multiplication_table(
    alphabet: tuple[str, ...],
    degree: int,
    basis: tuple[tuple[str, ...], ...],
    completion: GroebnerShirshovResult,
    reducible_pairs: frozenset[tuple[int, int]],
) -> tuple[tuple[FreeAlgebraPolynomial, ...], ...]:
    zero = FreeAlgebraPolynomial.model_construct(alphabet=alphabet, terms=())
    multiplication = []
    dimension = len(basis)
    for left_index, left_word in enumerate(basis):
        row = []
        for right_index, right_word in enumerate(basis):
            product_word = left_word + right_word
            if len(product_word) > degree:
                row.append(zero)
            elif (left_index, right_index) in reducible_pairs:
                monomial = _encode(alphabet, {product_word: Fraction(1)})
                row.append(_normal_form(monomial, completion.basis, degree))
            else:
                row.append(_encode(alphabet, {product_word: Fraction(1)}))
            if (left_index * max(1, dimension) + right_index) % 4_096 == 0:
                request_checkpoint("during truncated quotient multiplication table")
        multiplication.append(tuple(row))
    return tuple(multiplication)


def truncated_quotient_algebra(
    ideal: FreeAlgebraIdeal | Mapping[str, Any], degree: int
) -> TruncatedFreeAlgebraQuotient:
    """Return the exact finite algebra ``QQ<X>/(I + F_{>D})``.

    The ideal must be two-sided and homogeneous. Products above D are zero;
    other basis products reduce through the complete degree-D Groebner-Shirshov
    basis.
    """
    value = _as_ideal(ideal)
    if value.side != "two-sided":
        raise OperationDomainValidationError(
            location=("ideal", "side"),
            code="free_algebra.truncated_quotient_requires_two_sided",
            message="a truncated quotient requires side='two-sided'",
        )
    if (
        not isinstance(degree, int)
        or isinstance(degree, bool)
        or not 0 <= degree <= MAX_FREE_ALGEBRA_WORD_LENGTH
    ):
        _reject_resource(
            ("degree",),
            "truncated_quotient_degree_bound",
            "truncated quotient degree exceeds the admitted envelope",
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
    basis = _normal_words_through_degree(value, degree, leading_words)
    reducible_pairs = _admit_truncated_table(
        value, completion, degree, basis, leading_words
    )
    multiplication = _truncated_multiplication_table(
        value.alphabet, degree, basis, completion, reducible_pairs
    )
    zero = FreeAlgebraPolynomial.model_construct(alphabet=value.alphabet, terms=())
    unit = _encode(value.alphabet, {(): Fraction(1)}) if () in set(basis) else zero
    return TruncatedFreeAlgebraQuotient.model_construct(
        ideal=value,
        completion=completion,
        degree=degree,
        basis_words=basis,
        multiplication=multiplication,
        unit=unit,
    )


__all__ = [
    "compare_words",
    "concatenate_words",
    "factor_avoidance_dfa",
    "groebner_shirshov_through_degree",
    "ideal_degree_component",
    "ideal_generated_prefix",
    "ideal_membership",
    "multiply",
    "power_word",
    "quotient_normal_word_profile",
    "reverse_polynomial_antiautomorphism",
    "reverse_word",
    "substitute_word",
    "word_factors",
    "word_overlaps",
    "word_prefixes",
    "word_suffixes",
]


def power_polynomial(
    polynomial: FreeAlgebraPolynomial, exponent: int
) -> FreeAlgebraPolynomial:
    """Return a bounded nonnegative power using admitted squaring products."""

    value = _admit_polynomial(polynomial, label="polynomial")
    if (
        not isinstance(exponent, int)
        or isinstance(exponent, bool)
        or not 0 <= exponent <= MAX_FREE_ALGEBRA_POLYNOMIAL_POWER_EXPONENT
    ):
        _reject_resource(
            ("exponent",),
            "polynomial_power_exponent",
            "polynomial power exponent must be an integer from 0 through 64",
        )
    unit = FreeAlgebraPolynomial(
        alphabet=value.alphabet,
        terms=(
            FreeAlgebraTerm(
                coefficient=CanonicalRational.from_fraction(Fraction(1)), word=()
            ),
        ),
    )
    if exponent == 0:
        return unit
    if exponent == 1:
        return value
    if value.is_zero:
        return value

    result: FreeAlgebraPolynomial | None = None
    factor = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = factor if result is None else _multiply_for_power(result, factor)
        remaining >>= 1
        if remaining:
            factor = _multiply_for_power(factor, factor)
    return unit if result is None else result


def compose_polynomial_homomorphisms(
    f: FreeAlgebraPolynomialHomomorphism,
    g: FreeAlgebraPolynomialHomomorphism,
) -> FreeAlgebraPolynomialHomomorphism:
    """Return ``g ∘ f`` after aggregate admission of all generator images."""

    try:
        first = FreeAlgebraPolynomialHomomorphism.model_validate(f.model_dump())
        second = FreeAlgebraPolynomialHomomorphism.model_validate(g.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("homomorphisms",),
            code="free_algebra.homomorphism_composition_shape",
            message="both polynomial homomorphisms must be canonical",
        ) from exc
    if first.target_alphabet != second.source_alphabet:
        raise OperationDomainValidationError(
            location=("g", "source_alphabet"),
            code="free_algebra.homomorphism_composition_alphabet",
            message="f target alphabet must equal g source alphabet in the same order",
        )

    images, counts, lengths, letter_scalars = _admit_substitution_images(
        second, location_prefix=("g",)
    )
    for index, image in enumerate(first.images):
        if len(image.terms) > MAX_FREE_ALGEBRA_OPERAND_TERMS:
            _reject_resource(
                ("f", "images", index, "terms"),
                "substitution_operand_term_budget",
                "each generator image being composed is limited to 64 terms",
            )
        for term_index, term in enumerate(image.terms):
            if len(term.word) > MAX_FREE_ALGEBRA_WORD_LENGTH:
                _reject_resource(
                    ("f", "images", index, "terms", term_index, "word"),
                    "substitution_operand_word_length_budget",
                    "composed-map source image words are limited to 32 letters",
                )
    _preflight_substitution_expansion(
        second,
        first.images,
        images,
        counts,
        lengths,
        letter_scalars,
        source_locations=tuple(
            ("f", "images", i, "terms") for i in range(len(first.images))
        ),
    )
    composed_images = tuple(
        _expand_polynomial_substitution(second, image, images) for image in first.images
    )
    return FreeAlgebraPolynomialHomomorphism.model_construct(
        source_alphabet=first.source_alphabet,
        target_alphabet=second.target_alphabet,
        images=composed_images,
    )


def _multiply_for_power(
    left: FreeAlgebraPolynomial, right: FreeAlgebraPolynomial
) -> FreeAlgebraPolynomial:
    left, right = _admit_product(
        left,
        right,
        operand_term_limit=MAX_FREE_ALGEBRA_RESULT_TERMS,
        operand_word_length_limit=MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH,
    )
    # The operand limit admits words up to the result bound, but concatenation
    # adds their lengths. Without this preflight a 33-letter monomial squared
    # passes both operand checks, `multiply_sparse` expands it to 66 letters,
    # and FreeAlgebraTerm raises a pydantic ValidationError instead of the
    # operation returning a bounded resource refusal.
    result_word_length = max(
        (len(term.word) for term in left.terms),
        default=0,
    ) + max((len(term.word) for term in right.terms), default=0)
    if result_word_length > MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH:
        _reject_resource(
            ("polynomial",),
            "power_result_word_length_budget",
            (
                "the product's result word exceeds "
                f"{MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH} letters"
            ),
        )
    product, _ledger = multiply_sparse(left, right)
    return product
