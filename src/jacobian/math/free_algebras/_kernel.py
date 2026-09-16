"""Direct sparse noncommutative polynomial multiplication kernel.

The kernel performs one exact distributive concatenation product.  Semantic
admission has already established every operand and growth bound, so this
module computes only the product and its ledger and constructs the canonical
result value.  It never re-runs admission.
"""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian.math.free_algebras._models import (
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
    TermPairMultiplicationLedger,
    canonical_word_key,
)


def multiply_sparse(
    left: FreeAlgebraPolynomial,
    right: FreeAlgebraPolynomial,
) -> tuple[FreeAlgebraPolynomial, TermPairMultiplicationLedger]:
    """Return the exact distributive product and its bounded ledger.

    Every term pair ``(u, v)`` contributes ``a_u * b_v`` on the concatenated
    word ``uv``.  Like words are accumulated exactly and zero sums are
    dropped, so the returned polynomial is the unique canonical sparse form.
    """

    alphabet = left.alphabet
    accumulated: dict[tuple[str, ...], Fraction] = {}
    term_pair_count = 0
    collected_pair_count = 0
    for left_term in left.terms:
        left_coefficient = left_term.coefficient.as_fraction()
        for right_term in right.terms:
            term_pair_count += 1
            word = left_term.word + right_term.word
            contribution = right_term.coefficient.as_fraction() * left_coefficient
            if word in accumulated:
                collected_pair_count += 1
                accumulated[word] += contribution
            else:
                accumulated[word] = contribution

    zero_coefficient_word_count = sum(1 for value in accumulated.values() if value == 0)
    ordered = tuple(
        sorted(
            ((word, value) for word, value in accumulated.items() if value != 0),
            key=lambda item: canonical_word_key(alphabet, item[0]),
            reverse=True,
        )
    )
    terms = tuple(
        FreeAlgebraTerm(
            coefficient=CanonicalRational.from_fraction(value),
            word=word,
        )
        for word, value in ordered
    )
    product = FreeAlgebraPolynomial.model_construct(alphabet=alphabet, terms=terms)
    ledger = TermPairMultiplicationLedger(
        left_term_count=len(left.terms),
        right_term_count=len(right.terms),
        term_pair_count=term_pair_count,
        distinct_product_word_count=len(accumulated),
        collected_pair_count=collected_pair_count,
        zero_coefficient_word_count=zero_coefficient_word_count,
        result_term_count=len(terms),
        max_result_word_length=max((len(word) for word, _ in ordered), default=0),
        max_result_coefficient_digits=max(
            (canonical_rational_component_digits(term.coefficient) for term in terms),
            default=0,
        ),
    )
    return product, ledger


__all__ = ["multiply_sparse"]
