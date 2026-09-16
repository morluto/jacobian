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

from typing import Annotated, Self

from pydantic import AfterValidator, Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian._models import StrictModel

# The algebra's generator axis is interpretation-critical; a word's generator
# order is part of its identity.  These bounds describe the shared value
# representation, not one operation's resource envelope.
MAX_FREE_ALGEBRA_GENERATORS = 26
MAX_FREE_ALGEBRA_LETTER_LENGTH = 64
# A declared word (and therefore an operand term) is bounded by this length.
MAX_FREE_ALGEBRA_WORD_LENGTH = 32
# The product of two declared words has length at most twice that bound.
MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH = 2 * MAX_FREE_ALGEBRA_WORD_LENGTH
MAX_FREE_ALGEBRA_OPERAND_TERMS = 64
MAX_FREE_ALGEBRA_RESULT_TERMS = 4_096
MAX_FREE_ALGEBRA_TERM_PAIRS = MAX_FREE_ALGEBRA_OPERAND_TERMS**2
MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS = 64


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
        max_length=MAX_FREE_ALGEBRA_WORD_LENGTH,
        description=(
            "Ordered generator labels spelling one word; the empty tuple is the "
            "multiplicative unit."
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


__all__ = [
    "MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS",
    "MAX_FREE_ALGEBRA_GENERATORS",
    "MAX_FREE_ALGEBRA_LETTER_LENGTH",
    "MAX_FREE_ALGEBRA_OPERAND_TERMS",
    "MAX_FREE_ALGEBRA_RESULT_TERMS",
    "MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH",
    "MAX_FREE_ALGEBRA_TERM_PAIRS",
    "MAX_FREE_ALGEBRA_WORD_LENGTH",
    "FreeAlgebraLetter",
    "FreeAlgebraPolynomial",
    "FreeAlgebraPolynomialProductRequest",
    "FreeAlgebraPolynomialProductResult",
    "FreeAlgebraTerm",
    "FreeAlgebraWord",
    "TermPairMultiplicationLedger",
    "canonical_word_key",
]
