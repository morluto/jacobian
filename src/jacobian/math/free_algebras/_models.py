"""Typed contracts for exact free associative words and NC polynomials.

One free associative algebra ``K<X>`` over ``QQ`` is bound by one finite
ordered generator alphabet.  A word is a literal ordered tuple of generator
labels; the empty word is the multiplicative unit.  A noncommutative
polynomial is a sparse exact ``QQ``-linear combination of words, collected on
like words and stored in the canonical descending degree-lexicographic order
(by word length, then by generator rank).  The free algebra is genuinely
noncommutative, so ``x*y`` and ``y*x`` remain distinct words for distinct
generators ``x`` and ``y``.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian._models import StrictModel

# The algebra's generator axis is interpretation-critical; a word's generator
# order is part of its identity.  These bounds describe the shared value
# representation, not one operation's resource envelope.
MAX_FREE_ALGEBRA_GENERATORS = 26
MAX_FREE_ALGEBRA_LETTER_LENGTH = 64
# A growing-operation source word (and therefore an operand term) is bounded
# by this length.
MAX_FREE_ALGEBRA_WORD_LENGTH = 32
# The product of two declared words has length at most twice that bound.
MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH = 2 * MAX_FREE_ALGEBRA_WORD_LENGTH
# One canonical word value: results and non-growing single-word consumers may
# carry this many letters.
MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH = MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH
MAX_FREE_WORD_POWER_EXPONENT = 64
# Prefix/suffix/factor families are bounded over canonical word values, since
# those non-growing consumers admit producers through the full 64-letter range.
MAX_FREE_WORD_SPLITS = MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH + 1
MAX_FREE_WORD_SPLIT_LETTER_CELLS = (
    MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH * MAX_FREE_WORD_SPLITS
)
MAX_FREE_WORD_FACTOR_OCCURRENCES = (
    (MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH + 1)
    * (MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH + 2)
    // 2
)
MAX_FREE_WORD_FACTOR_LETTER_CELLS = (
    MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH
    * (MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH + 1)
    * (MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH + 2)
    // 6
)
MAX_FREE_WORD_FACTOR_DISTINCT = (
    MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH * (MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH + 1) // 2
    + 1
)
MAX_FREE_WORD_OVERLAP_ALIGNMENTS = 2 * MAX_FREE_ALGEBRA_WORD_LENGTH - 1
MAX_FREE_WORD_OVERLAP_LETTER_CELLS = 14_000
MAX_FREE_WORD_OVERLAP_WORK = 4_096
MAX_FREE_ALGEBRA_OPERAND_TERMS = 64
MAX_FREE_ALGEBRA_ADDITION_TERMS = 2 * MAX_FREE_ALGEBRA_OPERAND_TERMS
# Aggregate output allocation bounds count stored Unicode scalar cells and
# coefficient digit cells plus fixed per-record allowances.  They bound the
# canonical result's allocation, not any transport serialization.
MAX_FREE_ALGEBRA_ADDITION_OUTPUT_CELLS = 150_000
MAX_FREE_ALGEBRA_RESULT_TERMS = 4_096
MAX_FREE_ALGEBRA_TERM_PAIRS = MAX_FREE_ALGEBRA_OPERAND_TERMS**2
MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS = 64
MAX_FREE_ALGEBRA_SUBSTITUTION_EXPANSIONS = 65_536
MAX_FREE_ALGEBRA_SUBSTITUTION_WORK = 1_000_000
MAX_FREE_ALGEBRA_SUBSTITUTION_OUTPUT_CELLS = 150_000
# Ideal-prefix output is an aggregate carrier: unlike multiplication, it
# returns many basis polynomials at once. Keep that envelope independent from
# the per-polynomial result bound.
MAX_FREE_ALGEBRA_IDEAL_PREFIX_BASIS = MAX_FREE_ALGEBRA_RESULT_TERMS // 4
MAX_FREE_ALGEBRA_IDEAL_PREFIX_TOTAL_TERMS = MAX_FREE_ALGEBRA_RESULT_TERMS // 2
MAX_FREE_ALGEBRA_IDEAL_PREFIX_CELLS = 150_000
# A direct homogeneous ideal-component calculation is a dense exact row-space
# problem.  Keep its ambient word axis and returned basis small enough for a
# single request-scoped rational elimination.
MAX_FREE_ALGEBRA_IDEAL_COMPONENT_WORDS = 128
MAX_FREE_ALGEBRA_IDEAL_COMPONENT_CONTEXTS = 256
MAX_FREE_ALGEBRA_IDEAL_COMPONENT_MATRIX_CELLS = 32_768
MAX_FREE_ALGEBRA_IDEAL_COMPONENT_CELLS = 150_000
# GS completion checks every ordered basis pair at each fixed-point round.
# These are execution-envelope bounds; COMPLETE_THROUGH_DEGREE is returned only
# after the bounded rounds reach a genuine zero-composition fixed point.
MAX_FREE_ALGEBRA_GS_PAIR_CHECKS = MAX_FREE_ALGEBRA_TERM_PAIRS
MAX_FREE_ALGEBRA_GS_COMPOSITIONS = MAX_FREE_ALGEBRA_RESULT_TERMS
MAX_FREE_ALGEBRA_GS_REDUCTION_STEPS = (
    MAX_FREE_ALGEBRA_RESULT_TERMS * MAX_FREE_ALGEBRA_WORD_LENGTH
)
MAX_FREE_ALGEBRA_QUOTIENT_PROFILE_CANDIDATES = 16_384
MAX_FREE_ALGEBRA_QUOTIENT_PROFILE_OUTPUT_CELLS = 150_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by the free-algebra contracts."""

    return PydanticCustomError(f"free_algebra.{reason}", message)


def _require_unicode_scalar_string(value: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise _validation_error(
            "letter_not_unicode_scalar",
            "generator letter must contain only Unicode scalar values",
        )
    return value


FreeAlgebraLetter = Annotated[
    str,
    StringConstraints(
        min_length=1, max_length=MAX_FREE_ALGEBRA_LETTER_LENGTH, strict=True
    ),
    AfterValidator(_require_unicode_scalar_string),
]


def canonical_word_key(
    alphabet: tuple[str, ...], word: tuple[str, ...]
) -> tuple[int, tuple[int, ...]]:
    """Return the canonical degree-lexicographic key of one declared word.

    The key orders first by word length (total degree) and then by the tuple
    of generator ranks, so greater keys are greater words under the free
    algebra's admissible degree-lexicographic order.
    """

    return (len(word), tuple(alphabet.index(letter) for letter in word))


def _require_distinct_alphabet(alphabet: tuple[str, ...]) -> None:
    if len(set(alphabet)) != len(alphabet):
        raise _validation_error(
            "alphabet_letters_not_distinct",
            "generator alphabet letters must be distinct",
        )


class FreeAlgebraWord(StrictModel):
    """One bounded word over an explicitly ordered generator alphabet.

    The letters are literal generator labels; ``letters`` is the empty tuple
    for the multiplicative unit.  An empty alphabet carries only the empty
    word.
    """

    alphabet: tuple[FreeAlgebraLetter, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_GENERATORS,
        description=(
            "The declared ordered generator alphabet.  Its order fixes every "
            "generator rank used by the free algebra's word order."
        ),
    )
    letters: tuple[FreeAlgebraLetter, ...] = Field(
        default=(),
        max_length=MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH,
        description=(
            "Ordered generator labels spelling one word; the empty tuple is the "
            "multiplicative unit. Growing-operation sources admit at most 32 "
            "letters; canonical word values contain up to 64."
        ),
    )

    @model_validator(mode="after")
    def require_word_over_declared_alphabet(self) -> Self:
        _require_distinct_alphabet(self.alphabet)
        if any(letter not in self.alphabet for letter in self.letters):
            raise _validation_error(
                "word_letter_outside_alphabet",
                "word letter is outside the declared generator alphabet",
            )
        return self

    @property
    def length(self) -> int:
        """Total degree of this word."""

        return len(self.letters)

    @property
    def is_unit(self) -> bool:
        """Whether this word is the empty multiplicative unit."""

        return not self.letters


class FreeAlgebraWordPairRequest(StrictModel):
    """Two words over one ordered alphabet for a binary word operation."""

    left: FreeAlgebraWord = Field(description="Source word of at most 32 letters.")
    right: FreeAlgebraWord = Field(description="Source word of at most 32 letters.")

    @model_validator(mode="after")
    def require_shared_alphabet(self) -> Self:
        if self.left.alphabet != self.right.alphabet:
            raise _validation_error(
                "word_alphabet_mismatch", "words must use the same ordered alphabet"
            )
        if max(self.left.length, self.right.length) > MAX_FREE_ALGEBRA_WORD_LENGTH:
            raise _validation_error(
                "word_source_length",
                "source words may contain at most 32 letters",
            )
        return self


class FreeAlgebraWordPairResult(StrictModel):
    """Concatenation with source intervals and degree addition."""

    left: FreeAlgebraWord
    right: FreeAlgebraWord
    product: FreeAlgebraWord
    left_range: tuple[int, int]
    right_range: tuple[int, int]
    degree_addition: tuple[int, int, int]


class FreeAlgebraWordRequest(StrictModel):
    """One word subject to the 64-letter canonical value bound.

    Non-growing single-word consumers (reversal, prefix/suffix/factor
    families) admit producer words through the full canonical value range.
    """

    word: FreeAlgebraWord = Field(description="Word of at most 64 letters.")

    @model_validator(mode="after")
    def require_bounded_value(self) -> Self:
        if self.word.length > MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH:
            raise _validation_error(
                "word_value_length",
                "words may contain at most 64 letters",
            )
        return self


class FreeAlgebraWordPowerRequest(FreeAlgebraWordRequest):
    """One source word for a growing power, subject to the 32-letter bound."""

    exponent: int = Field(ge=0, le=MAX_FREE_WORD_POWER_EXPONENT)

    @model_validator(mode="after")
    def require_bounded_power_source(self) -> Self:
        if self.word.length > MAX_FREE_ALGEBRA_WORD_LENGTH:
            raise _validation_error(
                "word_source_length",
                "power source words may contain at most 32 letters",
            )
        return self


class FreeAlgebraWordPowerResult(StrictModel):
    word: FreeAlgebraWord
    exponent: int = Field(ge=0, le=MAX_FREE_WORD_POWER_EXPONENT)
    power: FreeAlgebraWord
    degree_multiplication: tuple[int, int, int]


class FreeAlgebraWordReverseResult(StrictModel):
    word: FreeAlgebraWord
    reverse: FreeAlgebraWord


class FreeAlgebraWordCompareResult(StrictModel):
    """Degree-lexicographic comparison: -1, 0, or 1."""

    left: FreeAlgebraWord
    right: FreeAlgebraWord
    comparison: Literal[-1, 0, 1]
    degree_comparison: Literal[-1, 0, 1]
    generator_order_comparison: Literal[-1, 0, 1]
    first_differing_position: int | None = Field(
        default=None, ge=0, le=MAX_FREE_ALGEBRA_WORD_LENGTH
    )
    left_generator_rank: int | None = Field(
        default=None, ge=0, le=MAX_FREE_ALGEBRA_GENERATORS - 1
    )
    right_generator_rank: int | None = Field(
        default=None, ge=0, le=MAX_FREE_ALGEBRA_GENERATORS - 1
    )


class FreeAlgebraWordPrefixSplit(StrictModel):
    """One prefix and its complementary suffix."""

    prefix_letters: tuple[FreeAlgebraLetter, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH
    )
    completing_suffix_letters: tuple[FreeAlgebraLetter, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH
    )


class FreeAlgebraWordSuffixSplit(StrictModel):
    """One suffix and its complementary prefix."""

    completing_prefix_letters: tuple[FreeAlgebraLetter, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH
    )
    suffix_letters: tuple[FreeAlgebraLetter, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH
    )


class FreeAlgebraWordPrefixesResult(StrictModel):
    word: FreeAlgebraWord
    splits: tuple[FreeAlgebraWordPrefixSplit, ...] = Field(
        max_length=MAX_FREE_WORD_SPLITS
    )


class FreeAlgebraWordSuffixesResult(StrictModel):
    word: FreeAlgebraWord
    splits: tuple[FreeAlgebraWordSuffixSplit, ...] = Field(
        max_length=MAX_FREE_WORD_SPLITS
    )


class FreeAlgebraWordFactorOccurrences(StrictModel):
    """One distinct factor and every start position where it occurs."""

    letters: tuple[FreeAlgebraLetter, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH
    )
    positions: tuple[int, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH + 1
    )


class FreeAlgebraWordFactorsResult(StrictModel):
    """Distinct contiguous factors with all source occurrence positions."""

    word: FreeAlgebraWord
    factors: tuple[FreeAlgebraWordFactorOccurrences, ...] = Field(
        max_length=MAX_FREE_WORD_FACTOR_DISTINCT
    )


class FreeAlgebraWordOverlapWitness(StrictModel):
    """One common word with both source occurrences and their contexts."""

    kind: Literal["OVERLAP", "INCLUSION"]
    common_word: FreeAlgebraWord
    left_offset: int = Field(ge=0, le=MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH)
    right_offset: int = Field(ge=0, le=MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH)
    left_prefix: tuple[FreeAlgebraLetter, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_WORD_LENGTH
    )
    left_suffix: tuple[FreeAlgebraLetter, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_WORD_LENGTH
    )
    right_prefix: tuple[FreeAlgebraLetter, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_WORD_LENGTH
    )
    right_suffix: tuple[FreeAlgebraLetter, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_WORD_LENGTH
    )


class FreeAlgebraWordOverlapResult(StrictModel):
    """All nontrivial overlap and proper inclusion ambiguities for two words."""

    left: FreeAlgebraWord
    right: FreeAlgebraWord
    witnesses: tuple[FreeAlgebraWordOverlapWitness, ...] = Field(
        max_length=MAX_FREE_WORD_OVERLAP_ALIGNMENTS
    )


class FreeAlgebraWordSubstitution(StrictModel):
    """A specified free-monoid homomorphism between two generator alphabets."""

    source_alphabet: tuple[FreeAlgebraLetter, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_GENERATORS
    )
    target_alphabet: tuple[FreeAlgebraLetter, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_GENERATORS
    )
    images: tuple[FreeAlgebraWord, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_GENERATORS,
        description="One target word of at most 32 letters for each source generator.",
    )

    @model_validator(mode="after")
    def require_generator_images(self) -> Self:
        _require_distinct_alphabet(self.source_alphabet)
        _require_distinct_alphabet(self.target_alphabet)
        if len(self.images) != len(self.source_alphabet):
            raise _validation_error(
                "substitution_image_count",
                "substitution needs one image per source generator",
            )
        if any(image.alphabet != self.target_alphabet for image in self.images):
            raise _validation_error(
                "substitution_target_alphabet",
                "every generator image must use the target alphabet",
            )
        if any(image.length > MAX_FREE_ALGEBRA_WORD_LENGTH for image in self.images):
            raise _validation_error(
                "substitution_image_length",
                "generator images may contain at most 32 letters",
            )
        return self


class FreeAlgebraWordSubstitutionRequest(StrictModel):
    substitution: FreeAlgebraWordSubstitution
    word: FreeAlgebraWord = Field(description="Source word of at most 32 letters.")

    @model_validator(mode="after")
    def require_source_word(self) -> Self:
        if self.word.length > MAX_FREE_ALGEBRA_WORD_LENGTH:
            raise _validation_error(
                "word_source_length", "source words may contain at most 32 letters"
            )
        if self.word.alphabet != self.substitution.source_alphabet:
            raise _validation_error(
                "substitution_source_alphabet",
                "word must use the substitution source alphabet",
            )
        return self


class FreeWordImageInterval(StrictModel):
    """Target half-open interval receiving one source letter occurrence."""

    source_index: int = Field(ge=0, lt=MAX_FREE_ALGEBRA_WORD_LENGTH)
    target_start: int = Field(ge=0, le=MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH)
    target_end: int = Field(ge=0, le=MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH)


class FreeAlgebraWordSubstitutionResult(StrictModel):
    substitution: FreeAlgebraWordSubstitution
    word: FreeAlgebraWord
    image: FreeAlgebraWord
    occurrence_images: tuple[FreeWordImageInterval, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_WORD_LENGTH
    )


class FreeAlgebraTerm(StrictModel):
    """One nonzero exact rational coefficient on a single declared word."""

    coefficient: CanonicalRational
    word: tuple[FreeAlgebraLetter, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH,
        description=(
            "Ordered generator labels of one word; the empty tuple is the "
            "multiplicative unit.  Zero coefficients are omitted canonically."
        ),
    )

    @model_validator(mode="after")
    def require_nonzero_bounded_term(self) -> Self:
        if self.coefficient.as_fraction() == 0:
            raise _validation_error("zero_term", "zero terms must be omitted")
        if (
            canonical_rational_component_digits(self.coefficient)
            > MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS
        ):
            raise _validation_error(
                "coefficient_digit_bound",
                "term coefficient exceeds the "
                f"{MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS}-digit bound",
            )
        return self


class FreeAlgebraPolynomial(StrictModel):
    """One canonical sparse exact ``QQ``-linear combination of free words.

    Terms are nonzero, collected on like words, and stored in descending
    canonical order (word length, then generator-rank lexicographic).  The
    empty support is the canonical zero polynomial and retains its alphabet.
    """

    alphabet: tuple[FreeAlgebraLetter, ...] = Field(
        default=(),
        max_length=MAX_FREE_ALGEBRA_GENERATORS,
        description=(
            "The declared ordered generator alphabet binding every word in the "
            "support and every future product."
        ),
    )
    terms: tuple[FreeAlgebraTerm, ...] = Field(
        default=(),
        max_length=MAX_FREE_ALGEBRA_RESULT_TERMS,
        description=(
            "Nonzero terms in descending canonical order: by word length, then "
            "lexicographically by generator rank.  Like words are collected."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_support(self) -> Self:
        _require_distinct_alphabet(self.alphabet)
        words = tuple(term.word for term in self.terms)
        if len(set(words)) != len(words):
            raise _validation_error(
                "duplicate_words", "like words must be collected before storage"
            )
        if any(letter not in self.alphabet for word in words for letter in word):
            raise _validation_error(
                "word_letter_outside_alphabet",
                "a polynomial word uses a letter outside the declared alphabet",
            )
        ordered = tuple(
            sorted(
                words,
                key=lambda word: canonical_word_key(self.alphabet, word),
                reverse=True,
            )
        )
        if words != ordered:
            raise _validation_error(
                "term_order",
                "polynomial terms must use descending length-lexicographic order",
            )
        return self

    @property
    def is_zero(self) -> bool:
        """Whether this polynomial has empty support."""

        return not self.terms


class TermPairMultiplicationLedger(StrictModel):
    """Bounded accounting for one distributive noncommutative product."""

    left_term_count: int = Field(ge=0, le=MAX_FREE_ALGEBRA_OPERAND_TERMS)
    right_term_count: int = Field(ge=0, le=MAX_FREE_ALGEBRA_OPERAND_TERMS)
    term_pair_count: int = Field(ge=0, le=MAX_FREE_ALGEBRA_TERM_PAIRS)
    distinct_product_word_count: int = Field(ge=0, le=MAX_FREE_ALGEBRA_RESULT_TERMS)
    collected_pair_count: int = Field(ge=0, le=MAX_FREE_ALGEBRA_TERM_PAIRS)
    zero_coefficient_word_count: int = Field(ge=0, le=MAX_FREE_ALGEBRA_TERM_PAIRS)
    result_term_count: int = Field(ge=0, le=MAX_FREE_ALGEBRA_RESULT_TERMS)
    max_result_word_length: int = Field(ge=0, le=MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH)
    max_result_coefficient_digits: int = Field(
        ge=0, le=MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS
    )


class FreeAlgebraPolynomialProductRequest(StrictModel):
    """Multiply two sparse NC polynomials over one shared generator alphabet."""

    left: FreeAlgebraPolynomial = Field(
        description=(
            "Left factor: a bounded exact sparse polynomial over its declared "
            "ordered alphabet.  Operand terms are limited to "
            f"{MAX_FREE_ALGEBRA_OPERAND_TERMS} and operand words to length "
            f"{MAX_FREE_ALGEBRA_WORD_LENGTH} by the product operation."
        )
    )
    right: FreeAlgebraPolynomial = Field(
        description=(
            "Right factor: bound to the identical ordered alphabet as ``left``."
        )
    )


class FreeAlgebraPolynomialAddRequest(StrictModel):
    """Add sparse NC polynomials over the identical ordered alphabet."""

    left: FreeAlgebraPolynomial
    right: FreeAlgebraPolynomial


class FreeAlgebraPolynomialProductResult(StrictModel):
    """Canonical distributive product with its multiplication ledger."""

    left: FreeAlgebraPolynomial
    right: FreeAlgebraPolynomial
    product: FreeAlgebraPolynomial
    ledger: TermPairMultiplicationLedger

    @model_validator(mode="after")
    def require_bound_canonical_product(self) -> Self:
        if self.left.alphabet != self.right.alphabet:
            raise _validation_error(
                "alphabet_binding",
                "both operands must be bound to the same ordered alphabet",
            )
        if self.product.alphabet != self.left.alphabet:
            raise _validation_error(
                "product_alphabet_binding",
                "the product must retain its operands' ordered alphabet",
            )
        ledger = self.ledger
        if (
            ledger.left_term_count != len(self.left.terms)
            or ledger.right_term_count != len(self.right.terms)
            or ledger.result_term_count != len(self.product.terms)
        ):
            raise _validation_error(
                "ledger_counts", "ledger term counts must match the bound operands"
            )
        if ledger.term_pair_count != ledger.left_term_count * ledger.right_term_count:
            raise _validation_error(
                "ledger_pair_count",
                "ledger term-pair count must equal the operand term product",
            )
        if (
            ledger.distinct_product_word_count > ledger.term_pair_count
            or ledger.collected_pair_count
            != ledger.term_pair_count - ledger.distinct_product_word_count
            or ledger.zero_coefficient_word_count > ledger.distinct_product_word_count
        ):
            raise _validation_error(
                "ledger_collection",
                "ledger collected-word accounting must partition the term pairs",
            )
        return self


class FreeAlgebraPolynomialHomomorphism(StrictModel):
    """A specified unital QQ-algebra map between two free algebras."""

    source_alphabet: tuple[FreeAlgebraLetter, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_GENERATORS
    )
    target_alphabet: tuple[FreeAlgebraLetter, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_GENERATORS
    )
    images: tuple[FreeAlgebraPolynomial, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_GENERATORS,
        description="One target polynomial image, possibly zero, per source generator.",
    )

    @model_validator(mode="after")
    def require_generator_images(self) -> Self:
        _require_distinct_alphabet(self.source_alphabet)
        _require_distinct_alphabet(self.target_alphabet)
        if len(self.images) != len(self.source_alphabet):
            raise _validation_error(
                "polynomial_substitution_image_count",
                "polynomial substitution needs one image per source generator",
            )
        if any(image.alphabet != self.target_alphabet for image in self.images):
            raise _validation_error(
                "polynomial_substitution_target_alphabet",
                "every generator image must use the target alphabet",
            )
        return self


class FreeAlgebraPolynomialSubstitutionRequest(StrictModel):
    substitution: FreeAlgebraPolynomialHomomorphism
    polynomial: FreeAlgebraPolynomial

    @model_validator(mode="after")
    def require_source_alphabet(self) -> Self:
        if self.polynomial.alphabet != self.substitution.source_alphabet:
            raise _validation_error(
                "polynomial_substitution_source_alphabet",
                "polynomial must use the substitution source alphabet",
            )
        return self


class FreeAlgebraIdeal(StrictModel):
    """A finitely generated left, right, or two-sided ideal presentation."""

    alphabet: tuple[FreeAlgebraLetter, ...] = Field(
        default=(), max_length=MAX_FREE_ALGEBRA_GENERATORS
    )
    generators: tuple[FreeAlgebraPolynomial, ...] = Field(max_length=32)
    side: Literal["left", "right", "two-sided"]

    @model_validator(mode="after")
    def require_parent(self) -> Self:
        _require_distinct_alphabet(self.alphabet)
        if any(generator.alphabet != self.alphabet for generator in self.generators):
            raise _validation_error(
                "ideal_alphabet", "ideal generators must share the declared alphabet"
            )
        return self


class FreeAlgebraIdealPrefixRequest(StrictModel):
    ideal: FreeAlgebraIdeal
    degree: int = Field(ge=0, le=MAX_FREE_ALGEBRA_WORD_LENGTH)


class FreeAlgebraIdealPrefixResult(StrictModel):
    ideal: FreeAlgebraIdeal
    degree: int
    basis: tuple[FreeAlgebraPolynomial, ...]

    @model_validator(mode="after")
    def require_result_parent(self) -> Self:
        if any(value.alphabet != self.ideal.alphabet for value in self.basis):
            raise _validation_error(
                "ideal_basis_alphabet",
                "ideal prefix basis must retain the ideal alphabet",
            )
        return self


class GroebnerShirshovRequest(StrictModel):
    ideal: FreeAlgebraIdeal
    degree: int = Field(ge=0, le=MAX_FREE_ALGEBRA_WORD_LENGTH)


class GroebnerShirshovResult(StrictModel):
    ideal: FreeAlgebraIdeal
    degree: int
    basis: tuple[FreeAlgebraPolynomial, ...]
    compositions: tuple[FreeAlgebraPolynomial, ...]
    status: Literal["COMPLETE_THROUGH_DEGREE"] = "COMPLETE_THROUGH_DEGREE"

    @model_validator(mode="after")
    def require_result_parent(self) -> Self:
        if any(
            value.alphabet != self.ideal.alphabet
            for value in (*self.basis, *self.compositions)
        ):
            raise _validation_error(
                "gs_basis_alphabet", "GS values must retain the ideal alphabet"
            )
        return self


class FreeAlgebraQuotientProfileRequest(StrictModel):
    """Request the bounded graded quotient word-basis profile."""

    ideal: FreeAlgebraIdeal
    degree: int = Field(ge=0, le=MAX_FREE_ALGEBRA_WORD_LENGTH)


class FreeAlgebraQuotientDegreeComponent(StrictModel):
    """Canonical irreducible words in one homogeneous quotient degree."""

    degree: int = Field(ge=0, le=MAX_FREE_ALGEBRA_WORD_LENGTH)
    normal_words: tuple[tuple[FreeAlgebraLetter, ...], ...] = Field(
        max_length=MAX_FREE_ALGEBRA_QUOTIENT_PROFILE_CANDIDATES
    )


class FreeAlgebraQuotientProfileResult(StrictModel):
    """Source-bound normal words and Hilbert-function values through D."""

    ideal: FreeAlgebraIdeal
    degree: int = Field(ge=0, le=MAX_FREE_ALGEBRA_WORD_LENGTH)
    leading_words: tuple[tuple[FreeAlgebraLetter, ...], ...] = Field(
        max_length=MAX_FREE_ALGEBRA_RESULT_TERMS
    )
    components: tuple[FreeAlgebraQuotientDegreeComponent, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_WORD_LENGTH + 1
    )
    hilbert_function: tuple[int, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_WORD_LENGTH + 1
    )

    @model_validator(mode="after")
    def require_canonical_profile(self) -> Self:
        if self.ideal.side != "two-sided":
            raise _validation_error(
                "quotient_profile_side", "quotient profile requires a two-sided ideal"
            )
        if (
            len(self.components) != self.degree + 1
            or len(self.hilbert_function) != self.degree + 1
        ):
            raise _validation_error(
                "quotient_profile_degree_axis",
                "quotient profile must contain each degree from zero through D",
            )
        if any(
            any(letter not in self.ideal.alphabet for letter in word)
            for word in self.leading_words
        ):
            raise _validation_error(
                "quotient_profile_leading_word_alphabet",
                "leading words must use the ideal alphabet",
            )
        keys = tuple(
            canonical_word_key(self.ideal.alphabet, word) for word in self.leading_words
        )
        if keys != tuple(sorted(set(keys))):
            raise _validation_error(
                "quotient_profile_leading_words",
                "leading words must be distinct and in canonical ascending order",
            )
        if any(len(word) > self.degree for word in self.leading_words):
            raise _validation_error(
                "quotient_profile_leading_degree",
                "leading words must lie within the profile degree bound",
            )
        for degree, component in enumerate(self.components):
            if component.degree != degree:
                raise _validation_error(
                    "quotient_profile_component_degree",
                    "quotient components must be ordered by degree from zero",
                )
            if self.hilbert_function[degree] != len(component.normal_words):
                raise _validation_error(
                    "quotient_profile_dimension",
                    "Hilbert-function values must equal normal-word counts",
                )
            words = component.normal_words
            if any(
                any(letter not in self.ideal.alphabet for letter in word)
                for word in words
            ):
                raise _validation_error(
                    "quotient_profile_word_alphabet",
                    "normal words must use the ideal alphabet",
                )
            keys = tuple(
                canonical_word_key(self.ideal.alphabet, word) for word in words
            )
            if any(len(word) != degree for word in words) or keys != tuple(
                sorted(set(keys))
            ):
                raise _validation_error(
                    "quotient_profile_normal_words",
                    "normal words must be distinct, degree-homogeneous, and canonically ordered",
                )
            if any(
                any(
                    word[start : start + len(leading)] == leading
                    for leading in self.leading_words
                    for start in range(len(word) - len(leading) + 1)
                )
                for word in words
            ):
                raise _validation_error(
                    "quotient_profile_reducible_word",
                    "normal-word basis elements must avoid every leading-word factor",
                )
        return self


class FreeAlgebraIdealDegreeComponentRequest(StrictModel):
    """Request one homogeneous degree component of a two-sided ideal."""

    ideal: FreeAlgebraIdeal
    degree: int = Field(ge=0, le=MAX_FREE_ALGEBRA_WORD_LENGTH)


class FreeAlgebraIdealDegreeComponentResult(StrictModel):
    """Canonical row-space basis of one homogeneous ideal component."""

    ideal: FreeAlgebraIdeal
    degree: int = Field(ge=0, le=MAX_FREE_ALGEBRA_WORD_LENGTH)
    component_basis: tuple[FreeAlgebraPolynomial, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_IDEAL_COMPONENT_WORDS
    )
    ambient_dimension: int = Field(ge=0, le=MAX_FREE_ALGEBRA_IDEAL_COMPONENT_WORDS)
    ideal_dimension: int = Field(ge=0, le=MAX_FREE_ALGEBRA_IDEAL_COMPONENT_WORDS)

    @model_validator(mode="after")
    def require_result_contract(self) -> Self:
        if self.ideal.side != "two-sided":
            raise _validation_error(
                "component_side", "ideal component result must be two-sided"
            )
        if any(value.alphabet != self.ideal.alphabet for value in self.component_basis):
            raise _validation_error(
                "component_alphabet", "ideal component values must retain the alphabet"
            )
        if any(
            not polynomial.terms
            or any(len(term.word) != self.degree for term in polynomial.terms)
            for polynomial in self.component_basis
        ):
            raise _validation_error(
                "component_degree",
                "component basis polynomials must have the result degree",
            )
        if self.ambient_dimension != len(self.ideal.alphabet) ** self.degree:
            raise _validation_error(
                "component_ambient_dimension",
                "ambient dimension must match the degree word space",
            )
        if self.ideal_dimension > self.ambient_dimension:
            raise _validation_error(
                "component_dimension", "ideal dimension cannot exceed ambient dimension"
            )
        if self.ideal_dimension != len(self.component_basis):
            raise _validation_error(
                "component_dimension", "ideal dimension must equal the basis size"
            )
        return self


class FreeAlgebraIdealMembershipRequest(StrictModel):
    """Ask whether one bounded polynomial lies in a presented two-sided ideal."""

    ideal: FreeAlgebraIdeal
    polynomial: FreeAlgebraPolynomial


class FreeAlgebraIdealMembershipResult(StrictModel):
    """Exact bounded membership decision, or UNKNOWN if completion was not reached."""

    ideal: FreeAlgebraIdeal
    polynomial: FreeAlgebraPolynomial
    status: Literal["MEMBER", "NOT_MEMBER", "UNKNOWN"]
    normal_form: FreeAlgebraPolynomial | None = None
    completion_degree: int | None = Field(
        default=None, ge=0, le=MAX_FREE_ALGEBRA_WORD_LENGTH
    )

    @model_validator(mode="after")
    def require_decision_contract(self) -> Self:
        if self.ideal.side != "two-sided":
            raise _validation_error(
                "membership_side", "membership result requires a two-sided ideal"
            )
        if self.polynomial.alphabet != self.ideal.alphabet:
            raise _validation_error(
                "membership_alphabet", "membership values must share the ideal alphabet"
            )
        if (
            self.normal_form is not None
            and self.normal_form.alphabet != self.ideal.alphabet
        ):
            raise _validation_error(
                "membership_normal_form_alphabet",
                "normal form must retain the ideal alphabet",
            )
        if self.status == "UNKNOWN":
            if self.normal_form is not None or self.completion_degree is not None:
                raise _validation_error(
                    "membership_unknown_claim",
                    "UNKNOWN cannot carry a normal form or completion claim",
                )
        elif self.normal_form is None or self.completion_degree is None:
            raise _validation_error(
                "membership_missing_evidence",
                "a decision requires a completed normal form",
            )
        elif (
            max((len(term.word) for term in self.polynomial.terms), default=0)
            > self.completion_degree
        ):
            raise _validation_error(
                "membership_completion_degree",
                "completion degree must cover every candidate term",
            )
        elif (self.status == "MEMBER") != (not self.normal_form.terms):
            raise _validation_error(
                "membership_normal_form",
                "status must agree with whether the normal form is zero",
            )
        return self


# Review-facing canonical names; aliases preserve one carrier per mathematical value.
FreeWord = FreeAlgebraWord
NCPolynomial = FreeAlgebraPolynomial
LeftIdealPresentation = FreeAlgebraIdeal
RightIdealPresentation = FreeAlgebraIdeal
TwoSidedIdealPresentation = FreeAlgebraIdeal
DegreeBoundedGroebnerShirshovBasis = GroebnerShirshovResult

__all__ = [
    "MAX_FREE_ALGEBRA_ADDITION_OUTPUT_CELLS",
    "MAX_FREE_ALGEBRA_ADDITION_TERMS",
    "MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS",
    "MAX_FREE_ALGEBRA_GENERATORS",
    "MAX_FREE_ALGEBRA_GS_COMPOSITIONS",
    "MAX_FREE_ALGEBRA_GS_PAIR_CHECKS",
    "MAX_FREE_ALGEBRA_GS_REDUCTION_STEPS",
    "MAX_FREE_ALGEBRA_LETTER_LENGTH",
    "MAX_FREE_ALGEBRA_OPERAND_TERMS",
    "MAX_FREE_ALGEBRA_QUOTIENT_PROFILE_CANDIDATES",
    "MAX_FREE_ALGEBRA_QUOTIENT_PROFILE_OUTPUT_CELLS",
    "MAX_FREE_ALGEBRA_RESULT_TERMS",
    "MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH",
    "MAX_FREE_ALGEBRA_TERM_PAIRS",
    "MAX_FREE_ALGEBRA_WORD_LENGTH",
    "MAX_FREE_ALGEBRA_WORD_VALUE_LENGTH",
    "MAX_FREE_WORD_FACTOR_DISTINCT",
    "MAX_FREE_WORD_FACTOR_LETTER_CELLS",
    "MAX_FREE_WORD_FACTOR_OCCURRENCES",
    "MAX_FREE_WORD_OVERLAP_ALIGNMENTS",
    "MAX_FREE_WORD_OVERLAP_LETTER_CELLS",
    "MAX_FREE_WORD_POWER_EXPONENT",
    "DegreeBoundedGroebnerShirshovBasis",
    "FreeAlgebraIdeal",
    "FreeAlgebraIdealPrefixRequest",
    "FreeAlgebraIdealPrefixResult",
    "FreeAlgebraLetter",
    "FreeAlgebraPolynomial",
    "FreeAlgebraPolynomialAddRequest",
    "FreeAlgebraPolynomialProductRequest",
    "FreeAlgebraPolynomialProductResult",
    "FreeAlgebraQuotientDegreeComponent",
    "FreeAlgebraQuotientProfileRequest",
    "FreeAlgebraQuotientProfileResult",
    "FreeAlgebraTerm",
    "FreeAlgebraWord",
    "FreeAlgebraWordCompareResult",
    "FreeAlgebraWordFactorOccurrences",
    "FreeAlgebraWordFactorsResult",
    "FreeAlgebraWordOverlapResult",
    "FreeAlgebraWordOverlapWitness",
    "FreeAlgebraWordPairRequest",
    "FreeAlgebraWordPairResult",
    "FreeAlgebraWordPowerRequest",
    "FreeAlgebraWordPowerResult",
    "FreeAlgebraWordPrefixSplit",
    "FreeAlgebraWordPrefixesResult",
    "FreeAlgebraWordRequest",
    "FreeAlgebraWordReverseResult",
    "FreeAlgebraWordSubstitution",
    "FreeAlgebraWordSubstitutionRequest",
    "FreeAlgebraWordSubstitutionResult",
    "FreeAlgebraWordSuffixSplit",
    "FreeAlgebraWordSuffixesResult",
    "FreeWord",
    "FreeWordImageInterval",
    "GroebnerShirshovRequest",
    "GroebnerShirshovResult",
    "LeftIdealPresentation",
    "NCPolynomial",
    "RightIdealPresentation",
    "TermPairMultiplicationLedger",
    "TwoSidedIdealPresentation",
    "canonical_word_key",
]
