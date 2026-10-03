"""Admission and contract regressions for the free-algebra owner.

Each test fails against the corresponding defect and passes on the repair; the
negative control is this file run against unmodified `main`.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.free_algebras import (
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
    compare_words,
    power_word,
    reverse_polynomial_antiautomorphism,
)
from jacobian.math.free_algebras import operations as fa_operations
from jacobian.math.free_algebras._models import (
    MAX_FREE_ALGEBRA_ANTIAUTOMORPHISM_OUTPUT_CELLS,
    MAX_FREE_ALGEBRA_ANTIAUTOMORPHISM_WORK,
    MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS,
    FreeAlgebraCanonicalWordPairRequest,
    FreeAlgebraWord,
    FreeAlgebraWordPairRequest,
    canonical_word_key,
)
from jacobian.math.free_algebras._tools import TOOLS
from jacobian.math.free_algebras.commutator.operations import commutator
from jacobian.math.free_algebras.operations import add


def _polynomial(
    alphabet: tuple[str, ...],
    terms: dict[tuple[str, ...], Fraction | int],
) -> FreeAlgebraPolynomial:
    return FreeAlgebraPolynomial(
        alphabet=alphabet,
        terms=tuple(
            FreeAlgebraTerm(
                coefficient=CanonicalRational.from_fraction(Fraction(value)), word=word
            )
            for word, value in sorted(
                terms.items(),
                key=lambda item: canonical_word_key(alphabet, item[0]),
                reverse=True,
            )
            if value
        ),
    )


def _alphabetical(alphabet: tuple[str, ...], word: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(alphabet.index(letter) for letter in word)


def _digits(index: int, width: int) -> tuple[int, ...]:
    return tuple((index // (10**power)) % 10 for power in range(width - 1, -1, -1))


def _letters(*words: tuple[str, ...]) -> int:
    return sum(len(word) for word in words)


def test_identical_operands_skip_the_convolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[f, f] is zero, and the preflight already decided that plan.

    Convolving anyway runs 64x64 = 4,096 pair iterations, twice the 2,048
    ceiling the term and work bounds impose on any admitted non-trivial
    request, and discards the result.
    """
    reads: list[int] = []

    def counting_as_fraction(self: CanonicalRational) -> Fraction:
        reads.append(1)
        return Fraction(self.num, self.den)

    dense = _polynomial(
        ("x", "y"),
        dict.fromkeys(_sixty_four_distinct_words(), 1),
    )
    assert len(dense.terms) == 64
    monkeypatch.setattr(CanonicalRational, "as_fraction", counting_as_fraction)

    result = commutator(dense, dense)

    assert result.commutator.terms == ()
    # Re-validating the two operands reads 64 coefficients each. The
    # convolution would read every coefficient of both operands twice per pair,
    # which is 2 * 64 * 64 = 8,192 further reads.
    assert len(reads) == 2 * len(dense.terms)


def _sixty_four_distinct_words() -> list[tuple[str, ...]]:
    """64 distinct words over two generators, three to six letters long."""
    words: list[tuple[str, ...]] = []
    for length in (3, 4, 5, 6):
        for letters in product("xy", repeat=length):
            words.append(letters)
    return words[:64]


def test_forced_same_word_cancellation_is_not_charged_a_coefficient() -> None:
    """Distinct operands whose concatenations coincide cancel exactly."""
    left = _polynomial(("x",), {("x",): 10**63})
    right = _polynomial(("x",), {("x",): 10**63 - 1})

    result = commutator(left, right)

    assert result.commutator.terms == ()


def test_a_genuine_oversized_contribution_is_still_refused() -> None:
    left = _polynomial(("x", "y"), {("x",): 10**63})
    right = _polynomial(("x", "y"), {("y",): 10**63})

    with pytest.raises(OperationResourceAdmissionError) as error:
        commutator(left, right)

    assert (
        error.value.errors()[0]["type"] == "free_algebra.commutator.coefficient_growth"
    )


def test_word_comparison_admits_the_canonical_value_range() -> None:
    """A 64-letter word produced by a power must compare through the catalog.

    ``compare_words`` is a non-growing consumer; the published pair request
    applied the 32-letter *source* bound, so a producer's own output could not
    be compared.
    """
    source = FreeAlgebraWord(alphabet=("x",), letters=("x",))
    powered = power_word(source, 64).power

    assert powered.letters == ("x",) * 64
    assert compare_words(powered, source).comparison == 1

    request = FreeAlgebraCanonicalWordPairRequest(left=powered, right=source)
    assert request.left.length == 64
    with pytest.raises(ValueError) as exc_info:
        FreeAlgebraCanonicalWordPairRequest(
            left=source.model_copy(update={"letters": ("x",) * 65}),
            right=source,
        )
    assert exc_info.value.errors()[0]["type"] == "free_algebra.word_value_length"
    # The concatenating operations keep the source-word bound.
    with pytest.raises(ValueError) as exc_info:
        FreeAlgebraWordPairRequest(
            left=source.model_copy(update={"letters": ("x",) * 33}), right=source
        )
    assert exc_info.value.errors()[0]["type"] == "free_algebra.word_source_length"


def test_word_comparison_request_is_the_non_growing_pair() -> None:
    tool = next(
        item for item in TOOLS if item.operation_id == "free_word.order.compare"
    )

    assert tool.request_type is FreeAlgebraCanonicalWordPairRequest


def test_addition_description_advertises_the_value_word_bound() -> None:
    """Multiplication results are 64-letter words and addition admits them."""
    description = next(
        item.description
        for item in TOOLS
        if item.operation_id == "free_algebra.polynomial.add.compute"
    )

    assert "word length at most 32" not in description
    assert "word length at most 64" in description
    long_word = _polynomial(("x",), {("x",) * 64: 1})
    zero = _polynomial(("x",), {})
    assert add(long_word, zero).terms == long_word.terms


def test_antiautomorphism_charges_the_kernel_it_actually_runs() -> None:
    """Reversal does one rank lookup per cell, not a scan per alphabet letter.

    The 4,096-term, nine-letter polynomial over ten generators is inside the
    output-cell envelope and reverses in milliseconds, but the estimate charged
    every word character times the alphabet width and refused it.
    """
    alphabet = tuple("abcdefghij")
    words = sorted(
        {
            tuple(alphabet[digit] for digit in _digits(index, 4)) + ("j",) * 5
            for index in range(4_096)
        },
        key=lambda word: _alphabetical(alphabet, word),
        reverse=True,
    )
    assert len(words) == 4_096
    value = _polynomial(alphabet, dict.fromkeys(words, 1))
    admitted = fa_operations._antiautomorphism_admission_work(value)
    assert admitted < MAX_FREE_ALGEBRA_ANTIAUTOMORPHISM_WORK
    cells = _letters(*[term.word for term in value.terms])
    assert cells * 10 + 6 * cells < MAX_FREE_ALGEBRA_ANTIAUTOMORPHISM_WORK

    result = reverse_polynomial_antiautomorphism(value)

    assert len(result.terms) == len(value.terms)


def test_antiautomorphism_still_refuses_a_work_heavy_polynomial() -> None:
    """The per-cell charge keeps the previously refused shape refused."""
    word = ("x",) * 64
    term = FreeAlgebraTerm.model_construct(
        coefficient=CanonicalRational.from_fraction(Fraction(1)), word=word
    )
    oversized = FreeAlgebraPolynomial.model_construct(
        alphabet=("x",), terms=(term,) * 4_096
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        reverse_polynomial_antiautomorphism(oversized)

    assert (
        error.value.errors()[0]["type"] == "free_algebra.antiautomorphism_work_budget"
    )
    assert (
        fa_operations._antiautomorphism_admission_work(oversized)
        > MAX_FREE_ALGEBRA_ANTIAUTOMORPHISM_WORK
    )
    assert MAX_FREE_ALGEBRA_ANTIAUTOMORPHISM_OUTPUT_CELLS > 4_096 * 64
    assert MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS == 64
